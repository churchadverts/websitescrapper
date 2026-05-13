import json
import os
import re
import requests
import warnings
from bs4 import BeautifulSoup
from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from datetime import datetime, timezone

warnings.filterwarnings('ignore', message='Unverified HTTPS request')
load_dotenv()

# ==========================================
# 1. CONFIGURATION
# ==========================================
openai_key   = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

client_ai : OpenAI  = OpenAI(api_key=openai_key)
supabase  : Client  = create_client(supabase_url, supabase_key)

OPENAI_MODEL   = "gpt-4.1-mini"
MAX_PAGE_CHARS = 60000   # gpt-4.1-mini has large context — we can afford more

playwright_available = False
try:
    from playwright.sync_api import sync_playwright
    playwright_available = True
except ImportError:
    pass

# ==========================================
# 2. SUPABASE HELPERS
# ==========================================
def update_scrape_status(business_id: str, status: str):
    try:
        payload = {"scrape_status": status}
        if status in ("done", "failed"):
            payload["scrape_last_run_at"] = datetime.now(timezone.utc).isoformat()
        supabase.table("businesses").update(payload).eq("business_id", business_id).execute()
        print(f"  [DB] scrape_status → {status}")
    except Exception as e:
        print(f"  [DB] Failed to update scrape_status: {e}")


def save_raw_page(business_id: str, page_url: str, page_type: str, raw_text: str):
    try:
        res = (
            supabase.table("raw_website_data")
            .select("version")
            .eq("business_id", business_id)
            .eq("page_type", page_type)
            .order("version", desc=True)
            .limit(1)
            .execute()
        )
        version = (res.data[0]["version"] + 1) if res.data else 1

        supabase.table("raw_website_data").insert({
            "business_id": business_id,
            "url":         page_url,
            "page_type":   page_type,
            "raw_text":    raw_text[:MAX_PAGE_CHARS],
            "version":     version
        }).execute()
        print(f"  [DB] Saved '{page_type}' page ({len(raw_text)} chars, v{version})")
    except Exception as e:
        print(f"  [DB] Failed to save raw page ({page_type}): {e}")


def save_business_knowledge(business_id: str, knowledge_md: str, profile_json: dict):
    try:
        supabase.table("businesses").update({
            "business_knowledge": knowledge_md,
            "ai_config": {"scraper_profile": profile_json}
        }).eq("business_id", business_id).execute()
        print(f"  [DB] Business knowledge saved for {business_id}")
    except Exception as e:
        print(f"  [DB] Failed to save business knowledge: {e}")


def fetch_bot_config(bot_id: str) -> dict:
    """
    Pull prompt + model settings from ai_bots_config.
    Returns empty dict on failure — callers fall back to hardcoded defaults.
    """
    try:
        res = (
            supabase.table("ai_bots_config")
            .select("prompt, model, temperature, max_tokens")
            .eq("bot_id", bot_id)
            .eq("is_active", True)
            .single()
            .execute()
        )
        if res.data:
            return res.data
    except Exception:
        pass
    print(f"  [Config] '{bot_id}' not found in ai_bots_config — using fallback")
    return {}

# ==========================================
# 3. HTML UTILITY FUNCTIONS
# ==========================================
def clean_soup_for_text(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove navigation, footers, cookie banners and other boilerplate."""
    for sel in [
        "script", "style", "noscript", "svg", "nav", "header",
        "footer", "aside", "iframe", "form",
        "[class*='cookie']", "[class*='popup']", "[class*='modal']",
        "[class*='banner']", "[id*='cookie']",   "[id*='popup']",
        "[class*='newsletter']", "[class*='subscribe']",
        "[class*='sidebar']",    "[class*='widget']"
    ]:
        for tag in soup.select(sel):
            tag.extract()
    return soup


def get_clean_soup(url: str, headers: dict) -> BeautifulSoup | None:
    try:
        r = requests.get(url, headers=headers, timeout=15, verify=False)
        if r.status_code == 200:
            return BeautifulSoup(r.text, 'html.parser')
    except Exception as e:
        print(f"    ✘ requests failed ({url}): {e}")
    return None


def render_page_html(url: str, headers: dict, timeout: int = 20000) -> BeautifulSoup | None:
    if not playwright_available:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page    = browser.new_page(user_agent=headers.get("User-Agent", ""))
            page.goto(url, timeout=timeout)
            page.wait_for_timeout(2500)
            html = page.content()
            browser.close()
            return BeautifulSoup(html, 'html.parser')
    except Exception as e:
        print(f"    ✘ Playwright failed ({url}): {e}")
        return None


def fetch_page(url: str, headers: dict) -> BeautifulSoup | None:
    soup = get_clean_soup(url, headers)
    if soup:
        return soup
    return render_page_html(url, headers)

# ==========================================
# 4. DATA EXTRACTION FUNCTIONS
# ==========================================
def format_price(raw_price) -> str:
    if not raw_price:
        return "0.00"
    try:
        price_str = re.sub(r'[^\d.]', '', str(raw_price))
        if not price_str:
            return str(raw_price)
        # Handle prices stored as integers without decimal (e.g. 699900 → 6,999.00)
        if len(price_str) >= 5 and '.' not in price_str:
            return "{:,.2f}".format(float(price_str) / 100)
        return "{:,.2f}".format(float(price_str))
    except (ValueError, TypeError):
        return str(raw_price)


def extract_meta_info(soup: BeautifulSoup) -> dict:
    """OG tags and meta description — present on nearly every site."""
    info = {}
    for name, attr, key in [
        ("description",         "name",     "meta_description"),
        ("og:title",            "property", "og_title"),
        ("og:description",      "property", "og_description"),
        ("og:site_name",        "property", "site_name"),
        ("twitter:description", "name",     "twitter_description"),
    ]:
        tag = soup.find("meta", attrs={attr: name})
        if tag:
            value = (tag.get("content") or "").strip()
            if value:
                info[key] = value

    title_tag = soup.find("title")
    if title_tag:
        info["page_title"] = title_tag.get_text(strip=True)

    return info


def extract_contact_info(soup: BeautifulSoup) -> dict:
    """
    Phone numbers, emails, and physical addresses.
    Uses Kenyan number patterns: 07xx, 01xx, +254.
    """
    text = soup.get_text()

    phone_re = re.compile(
        r'(\+254[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{3}'
        r'|0[17]\d{8}'
        r'|0\d{2}[\s\-]?\d{3}[\s\-]?\d{4})'
    )
    email_re = re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')
    address_re = re.compile(
        r'(P\.?O\.?\s*Box[\s\w,\.]+|'
        r'(?:Nairobi|Mombasa|Kisumu|Nakuru|Eldoret|Thika|Westlands|'
        r'Kilimani|Karen|CBD|Upperhill|Lavington|Hurlingham)[\w\s,\.\-]{5,80})',
        re.IGNORECASE
    )

    phones    = list(dict.fromkeys(phone_re.findall(text)))[:5]
    emails    = [
        e for e in dict.fromkeys(email_re.findall(text))
        if not any(x in e.lower() for x in ['example', 'youremail', 'domain', 'email.com'])
    ][:5]
    addresses = list(dict.fromkeys(m if isinstance(m, str) else m[0]
                                   for m in address_re.findall(text)))[:3]

    return {"phones": phones, "emails": emails, "addresses": addresses}


def extract_table_pricing(soup: BeautifulSoup) -> list:
    """
    Finds HTML tables that contain a pricing column.
    Very common on Kenyan agency, service, and portfolio sites.
    """
    results = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue

        headers = [cell.get_text(strip=True).lower()
                   for cell in rows[0].find_all(["th", "td"])]

        price_col = next(
            (i for i, h in enumerate(headers)
             if any(k in h for k in ["price", "cost", "rate", "amount", "fee", "ksh", "kes"])),
            None
        )
        name_col = next(
            (i for i, h in enumerate(headers)
             if any(k in h for k in ["service", "package", "product", "plan",
                                      "name", "item", "description", "details"])),
            0
        )

        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if not cells:
                continue
            name  = cells[name_col] if name_col < len(cells) else cells[0]
            price = (cells[price_col]
                     if price_col is not None and price_col < len(cells)
                     else "Contact for price")
            if name and 2 < len(name) < 200:
                results.append({"title": name, "price": price, "source": "table"})

    return results


def extract_pricing_cards(soup: BeautifulSoup) -> list:
    """
    CSS class-based card/tile detection for pricing sections and package blocks.
    Works on most modern Kenyan business sites using card layouts.
    """
    results  = []
    seen     = set()
    price_re = re.compile(
        r'(KSh|KES|Ksh|ksh|/=|per\s+month|monthly|annually|\d{1,3},\d{3})',
        re.IGNORECASE
    )

    for selector in [
        "[class*='pricing']", "[class*='package']", "[class*='plan']",
        "[class*='service-card']", "[class*='price-card']", "[class*='tier']",
        "[class*='offer']", "[class*='tariff']", "[class*='product-card']"
    ]:
        for card in soup.select(selector):
            text = card.get_text(separator=" ", strip=True)
            if not price_re.search(text) or not (20 < len(text) < 1200):
                continue

            heading = card.find(["h2", "h3", "h4", "strong"])
            title   = heading.get_text(strip=True) if heading else text[:60]

            price_match = re.search(r'(?:KSh|KES|Ksh)?\s*[\d,]+(?:\.\d{2})?', text)
            price       = price_match.group(0).strip() if price_match else "Contact for price"

            key = (title[:50], price)
            if key not in seen:
                results.append({"title": title, "price": price, "description": text[:300]})
                seen.add(key)

    return results[:15]


def extract_service_headings(soup: BeautifulSoup) -> list:
    """
    Pull h2/h3 headings with the paragraph that follows.
    Works on flat service/portfolio pages with no structured data.
    """
    services     = []
    skip_phrases = {
        "menu", "navigation", "footer", "header", "login", "sign up",
        "register", "home", "contact us", "about us", "follow us",
        "get in touch", "latest posts", "recent posts", "tags", "categories",
        "search", "subscribe", "newsletter", "our clients", "partners"
    }

    for tag in soup.find_all(["h2", "h3"]):
        text = tag.get_text(strip=True)
        if not text or not (3 < len(text) < 120):
            continue
        if any(p in text.lower() for p in skip_phrases):
            continue

        next_p = tag.find_next_sibling("p")
        desc   = next_p.get_text(strip=True)[:250] if next_p else ""

        services.append({"heading": text, "description": desc})

    return services[:25]


def parse_json_ld_products(soup: BeautifulSoup, base_url: str) -> list:
    """Schema.org structured data — Product and ItemList types."""
    candidates = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            payload = json.loads(script.string or script.text)
        except Exception:
            continue

        for obj in (payload if isinstance(payload, list) else [payload]):
            if not isinstance(obj, dict):
                continue
            if obj.get("@type") == "Product":
                title = obj.get("name")
                image = obj.get("image")
                if isinstance(image, list):
                    image = image[0]
                price = None
                if isinstance(obj.get("offers"), dict):
                    price = obj["offers"].get("price")
                if title:
                    candidates.append({
                        "title":     title.strip(),
                        "price":     format_price(price),
                        "image_url": urljoin(base_url, image) if image else None
                    })
            if obj.get("@type") in ["ItemList", "Collection"]:
                for item in obj.get("itemListElement", []):
                    product = item.get("item") or item
                    if isinstance(product, dict) and product.get("@type") == "Product":
                        title = product.get("name")
                        image = product.get("image")
                        if isinstance(image, list):
                            image = image[0]
                        price = (product.get("offers", {}).get("price")
                                 if isinstance(product.get("offers"), dict) else None)
                        if title:
                            candidates.append({
                                "title":     title.strip(),
                                "price":     format_price(price),
                                "image_url": urljoin(base_url, image) if image else None
                            })
    return candidates


def extract_custom_candidates(soup: BeautifulSoup, base_url: str) -> list:
    """
    Fallback: image + price pattern detection for ecommerce product grids.
    Handles sites without structured data.
    """
    candidates = []
    price_re   = re.compile(
        r'(KSh|KES|₨|/=|UGX|TZS|NGN|USD|EUR|£|\d{2,3}(?:[.,]\d{2,3})?)'
    )
    skip_src   = ['logo', 'banner', 'icon', 'sprite', 'avatar',
                  'facebook', 'twitter', 'instagram', 'whatsapp']

    for container in soup.find_all(["div", "li", "article", "section"]):
        img = container.find("img")
        if not img:
            continue
        text = container.get_text(separator=" ", strip=True)
        if not (20 < len(text) < 800):
            continue
        match = price_re.search(text)
        if not match:
            continue
        src = (img.get("src") or img.get("data-src")
               or img.get("data-lazy-src") or img.get("data-original"))
        if not src or any(x in src.lower() for x in skip_src):
            continue
        heading = container.find(["h1", "h2", "h3", "h4", "a"])
        title   = heading.get_text(strip=True) if heading else text[:60]
        candidates.append({
            "title":     title,
            "price":     format_price(match.group(0)),
            "image_url": urljoin(base_url, src)
        })

    seen   = set()
    unique = []
    for c in candidates:
        key = (c.get("image_url"), c.get("title"))
        if key not in seen:
            unique.append(c)
            seen.add(key)
    return unique

# ==========================================
# 5. THE HARVESTER
# ==========================================
TARGET_CATEGORIES = {
    "services":     ["services", "service", "what-we-do", "solutions", "offerings"],
    "shop":         ["shop", "store", "products", "collection", "catalogue"],
    "pricing":      ["pricing", "price", "plans", "packages", "rates", "tariff"],
    "about":        ["about", "story", "who-we-are", "about-us"],
    "faq":          ["faq", "faqs", "questions", "support", "help"],
    "contact":      ["contact", "reach-us", "location", "find-us", "get-in-touch"],
    "portfolio":    ["portfolio", "gallery", "work", "projects", "case-studies"],
    "testimonials": ["testimonials", "reviews", "clients", "what-clients-say"]
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def harvester(url: str, business_id: str) -> dict:
    print(f"\n--- Harvesting: {url} ---")

    products          = []
    custom_candidates = []
    service_headings  = []
    all_pages         = {}   # page_type → (page_url, text)

    # ── Shopify API ──────────────────────────────────────────────
    try:
        r = requests.get(
            url.rstrip("/") + "/products.json",
            headers=HEADERS, timeout=5, verify=False
        )
        if r.status_code == 200:
            for p in r.json().get("products", [])[:10]:
                products.append({
                    "title":     p.get("title"),
                    "price":     format_price((p.get("variants") or [{}])[0].get("price")),
                    "image_url": ((p.get("images") or [{}])[0].get("src"))
                })
            if products:
                print(f"  ✓ Shopify API: {len(products)} products")
    except Exception:
        pass

    # ── WooCommerce API ──────────────────────────────────────────
    if not products:
        try:
            r = requests.get(
                url.rstrip("/") + "/wp-json/wc/store/products",
                headers=HEADERS, timeout=5, verify=False
            )
            if r.status_code == 200:
                for p in r.json()[:10]:
                    products.append({
                        "title":     p.get("name"),
                        "price":     format_price((p.get("prices") or {}).get("price")),
                        "image_url": ((p.get("images") or [{}])[0].get("src"))
                    })
                if products:
                    print(f"  ✓ WooCommerce API: {len(products)} products")
        except Exception:
            pass

    # ── Homepage ─────────────────────────────────────────────────
    soup = fetch_page(url, HEADERS)
    if not soup:
        print("  ✗ Could not fetch homepage")
        return {
            "products": [], "custom_candidates": [], "text": "",
            "url": url, "contact_info": {}, "service_headings": [], "all_pages": {}
        }

    meta_info    = extract_meta_info(soup)
    contact_info = extract_contact_info(soup)

    cleaned_home = clean_soup_for_text(BeautifulSoup(str(soup), "html.parser"))
    homepage_text = (
        f"[META]\n{json.dumps(meta_info, ensure_ascii=False)}\n\n"
        f"[CONTACT FOUND ON PAGE]\n{json.dumps(contact_info, ensure_ascii=False)}\n\n"
        f"[HOMEPAGE TEXT]\n{cleaned_home.get_text(separator=' ', strip=True)}"
    )
    all_pages["homepage"] = (url, homepage_text)

    # ── Product/service extraction from homepage ─────────────────
    if not products:
        custom_candidates.extend(parse_json_ld_products(soup, url))
        custom_candidates.extend(extract_table_pricing(soup))
        custom_candidates.extend(extract_pricing_cards(soup))
        custom_candidates.extend(extract_custom_candidates(soup, url))
        service_headings.extend(extract_service_headings(soup))

        # Playwright fallback if static scrape found nothing
        if not custom_candidates and not service_headings and playwright_available:
            print("  Trying Playwright fallback for homepage...")
            js_soup = render_page_html(url, HEADERS)
            if js_soup:
                custom_candidates.extend(parse_json_ld_products(js_soup, url))
                custom_candidates.extend(extract_table_pricing(js_soup))
                custom_candidates.extend(extract_pricing_cards(js_soup))
                custom_candidates.extend(extract_custom_candidates(js_soup, url))
                service_headings.extend(extract_service_headings(js_soup))

    # ── Discover and scrape sub-pages ────────────────────────────
    found_links = {}
    for a in soup.find_all("a", href=True):
        link_text = a.get_text(strip=True).lower()
        link_href = a["href"].strip().lower()
        full_link = urljoin(url, a["href"])

        try:
            if urlparse(full_link).netloc != urlparse(url).netloc:
                continue
        except Exception:
            continue
        if full_link.rstrip("/") == url.rstrip("/"):
            continue

        for category, keywords in TARGET_CATEGORIES.items():
            if category not in found_links:
                if any(k in link_text or k in link_href for k in keywords):
                    found_links[category] = full_link

    print(f"  Sub-pages found: {list(found_links.keys())}")

    for category, link in found_links.items():
        sub_soup = fetch_page(link, HEADERS)
        if not sub_soup:
            continue

        if category in ("shop", "services", "pricing") and not products:
            custom_candidates.extend(parse_json_ld_products(sub_soup, link))
            custom_candidates.extend(extract_table_pricing(sub_soup))
            custom_candidates.extend(extract_pricing_cards(sub_soup))
            custom_candidates.extend(extract_custom_candidates(sub_soup, link))
        if category in ("services", "portfolio"):
            service_headings.extend(extract_service_headings(sub_soup))

        cleaned_sub = clean_soup_for_text(BeautifulSoup(str(sub_soup), "html.parser"))
        all_pages[category] = (link, cleaned_sub.get_text(separator=" ", strip=True))

    # ── Deduplicate candidates ───────────────────────────────────
    unique_candidates = []
    seen = set()
    for c in custom_candidates:
        key = (c.get("title", "")[:80], c.get("price", ""))
        if key[0] and key not in seen:
            unique_candidates.append(c)
            seen.add(key)

    combined_text = "\n\n".join(
        f"[{ptype.upper()} PAGE]\n{text}"
        for ptype, (_, text) in all_pages.items()
    )

    print(f"  Total pages scraped: {len(all_pages)}")
    print(f"  Products found: {len(products) or len(unique_candidates)}")

    return {
        "products":          products[:20],
        "custom_candidates": unique_candidates[:25],
        "text":              combined_text,
        "url":               url,
        "contact_info":      contact_info,
        "service_headings":  service_headings[:25],
        "all_pages":         all_pages
    }

# ==========================================
# 6. THE CHEF  (GPT-4.1-mini)
# ==========================================
CHEF_FALLBACK_PROMPT = """
You are a data architect extracting structured business context for a sales AI system.
You must return a valid JSON object only — no markdown fences, no preamble, no explanation.

First, determine business_type: "ecommerce" or "service".

If "ecommerce" include these fields:
- business_name, business_type, brand_voice, unique_selling_point
- shipping_policy, return_policy, payment_methods
- top_10_products: array of { title, price, image_url }

If "service" include these fields:
- business_name, business_type, brand_voice, unique_selling_point
- service_packages: array of { service_name, description, price }
- booking_process, service_areas

Both types must always include:
- contact_info: { phone, email, physical_location }
- faq_and_objections: array of { question, answer }
- social_proof: array of quotes or stats found on the site
- payment_methods: always look for M-Pesa — it is the dominant method in Kenya

Strict rules:
- Return ONLY the JSON object. Nothing else before or after it.
- If a price looks like 699900, it is likely stored as cents — format it as 6,999.00.
- Ignore all UI chrome: Cart, Login, Menu, Home, Search, Subscribe.
- Prefer content from services, pricing, and about pages over homepage text.
- If a field has no data, use null — do not invent data.
"""


def chef(harvested_data: dict) -> dict | None:
    print("  [Chef] Structuring raw data with GPT-4.1-mini...")

    config        = fetch_bot_config("scraper_data_cleaner")
    model         = config.get("model", OPENAI_MODEL)
    max_tokens    = config.get("max_tokens", 4096)
    temperature   = config.get("temperature", 0.2)
    system_prompt = config.get("prompt", CHEF_FALLBACK_PROMPT)

    user_content = (
        f"WEBSITE TEXT:\n{harvested_data['text'][:50000]}\n\n"
        f"CONTACT INFO DETECTED:\n{json.dumps(harvested_data.get('contact_info', {}))}\n\n"
        f"SERVICE HEADINGS:\n{json.dumps(harvested_data.get('service_headings', []))}\n\n"
    )
    if harvested_data.get("products"):
        user_content += f"STRUCTURED PRODUCTS (from API):\n{json.dumps(harvested_data['products'])}"
    else:
        user_content += f"EXTRACTED CANDIDATES (from HTML):\n{json.dumps(harvested_data.get('custom_candidates', []))}"

    try:
        response = client_ai.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},   # forces clean JSON, no fences
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content}
            ]
        )
        profile = json.loads(response.choices[0].message.content)

        # If Shopify/WooCommerce returned structured products, use those directly
        if profile.get("business_type") == "ecommerce" and harvested_data.get("products"):
            profile["top_10_products"] = harvested_data["products"]

        print("  [Chef] ✓ Profile structured")
        return profile

    except Exception as e:
        print(f"  [Chef] ✗ {e}")
        return None

# ==========================================
# 7. THE ARCHITECT  (GPT-4.1-mini)
# ==========================================
ARCHITECT_FALLBACK_PROMPT = """
You are a business knowledge architect. Your job is to transform a structured business
JSON profile and raw website text into a 'Business Essence' document.

This document will be the complete knowledge base for an AI agent that handles
customer conversations and sales for this business. Write it as a manual for
a perfect employee — not for a developer. Never mention AI, prompts, or chatbots.

DOCUMENT STRUCTURE:

1. BUSINESS IDENTITY & ESSENCE
   - Core Role: one sentence describing the professional persona
   - Brand Voice: specific tone, language style, Kenyan market nuances
   - Competitive Edge: the specific USPs extracted from the data — no generic claims

2. LOGIC-DRIVEN SALES WORKFLOW
   Use If/Then logic and 'Agent Must' directives throughout.
   - PHASE 1: Hook & Qualify — how to open conversations, how to identify real leads
   - PHASE 2: Discovery — the exact data points to collect before making a recommendation
   - PHASE 3: The Pitch — how to present the solution based on what was discovered
   - PHASE 4: Objection Handling — specific rebuttals using this business's actual facts
     (price, trust, timeline, delivery — cover all common ones with business-specific answers)
   - PHASE 5: The Close — the criteria for a close and the exact call-to-action to use

3. BUSINESS GUARDRAILS
   - Things the agent must never say, promise, or claim
   - Specific situations that require human handover

4. PRODUCT / SERVICE CATALOGUE
   - Organised by category with descriptions and pricing
   - Payment methods with M-Pesa instructions if available
   - Delivery timelines, booking windows, service area limits

5. CONTACTS & TRUST SIGNALS
   - All phone numbers, emails, physical addresses found
   - Testimonials, repeat customer references, certifications, notable clients

Strict rules:
- Reference specific Kenyan context: city names, M-Pesa, local phrases, local norms
- Be specific throughout — a generic output is a failed output
- Never write 'the AI' or 'the chatbot' — write 'the agent' or just use directives
"""


def architect(profile: dict, raw_text: str) -> str | None:
    print("  [Architect] Building knowledge document with GPT-4.1-mini...")

    config        = fetch_bot_config("scraper_knowledge_builder")
    model         = config.get("model", OPENAI_MODEL)
    max_tokens    = config.get("max_tokens", 4096)
    temperature   = config.get("temperature", 0.3)
    system_prompt = config.get("prompt", ARCHITECT_FALLBACK_PROMPT)

    user_content = (
        f"JSON PROFILE:\n{json.dumps(profile, ensure_ascii=False)}\n\n"
        f"RAW SCRAPED TEXT (first 40,000 chars):\n{raw_text[:40000]}"
    )

    try:
        response = client_ai.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content}
            ]
        )
        result = response.choices[0].message.content
        print("  [Architect] ✓ Knowledge document built")
        return result

    except Exception as e:
        print(f"  [Architect] ✗ {e}")
        return None

# ==========================================
# 8. FASTAPI APP
# ==========================================
app = FastAPI(title="Business Scraper Service")


class OnboardingRequest(BaseModel):
    url:         str
    business_id: str


@app.post("/generate-profile")
def generate_business_profile(req: OnboardingRequest):
    business_id = req.business_id
    url         = req.url.strip()

    try:
        # 0. Mark running
        update_scrape_status(business_id, "running")
        print(f"\n[START] business_id={business_id}  url={url}")

        # 1. Harvest raw data
        raw_info = harvester(url, business_id)

        # 2. Save each page to raw_website_data
        for page_type, (page_url, page_text) in raw_info.get("all_pages", {}).items():
            save_raw_page(business_id, page_url, page_type, page_text)

        # 3. Chef — structure into JSON profile
        final_profile = chef(raw_info)
        if not final_profile:
            update_scrape_status(business_id, "failed")
            raise HTTPException(status_code=500, detail="Chef failed to structure data.")

        # 4. Architect — build knowledge document
        knowledge_md = architect(final_profile, raw_info["text"])
        if not knowledge_md:
            update_scrape_status(business_id, "failed")
            raise HTTPException(status_code=500, detail="Architect failed to build knowledge.")

        # 5. Save to businesses table
        save_business_knowledge(business_id, knowledge_md, final_profile)
        update_scrape_status(business_id, "done")

        print(f"[DONE] business_id={business_id}\n")

        return {
            "business_id":     business_id,
            "name":            final_profile.get("business_name", "Unknown"),
            "profile_json":    final_profile,
            "instructions_md": knowledge_md,
            "products":        raw_info.get("products") or raw_info.get("custom_candidates", []),
            "pages_scraped":   list(raw_info.get("all_pages", {}).keys()),
            "contact_info":    raw_info.get("contact_info", {})
        }

    except HTTPException:
        raise
    except Exception as e:
        update_scrape_status(business_id, "failed")
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/scrape-status/{business_id}")
def get_scrape_status(business_id: str):
    try:
        res = (
            supabase.table("businesses")
            .select("scrape_status, scrape_last_run_at, persona_pack_status")
            .eq("business_id", business_id)
            .single()
            .execute()
        )
        return res.data
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "Scraper service active"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
