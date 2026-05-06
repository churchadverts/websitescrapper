import json
import os
import re
import requests
from bs4 import BeautifulSoup
import openai
from dotenv import load_dotenv
from urllib.parse import urljoin, urlparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Optional Playwright support for JS-rendered sites
playwright_available = False
try:
    from playwright.sync_api import sync_playwright
    playwright_available = True
except ImportError:
    playwright_available = False

# ==========================================
# 1. CONFIGURATION
# ==========================================
client = openai.OpenAI(api_key=openai_key)

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def format_price(raw_price):
    if not raw_price:
        return "0.00"
    try:
        price_str = re.sub(r'[^\d.]', '', str(raw_price))
        if len(price_str) >= 4 and '.' not in price_str:
            return "{:.2f}".format(float(price_str) / 100)
        return "{:.2f}".format(float(price_str))
    except (ValueError, TypeError):
        return str(raw_price)

def get_clean_soup(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=15, verify=False)
        if response.status_code == 200:
            return BeautifulSoup(response.text, 'html.parser')
    except Exception as e:
        print(f"    ✘ Error accessing {url}: {e}")
    return None

def render_page_html(url, headers, wait_for=None, timeout=20000):
    if not playwright_available:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=headers.get('User-Agent'))
            page.goto(url, timeout=timeout)
            if wait_for:
                page.wait_for_selector(wait_for, timeout=timeout)
            page.wait_for_timeout(1000)
            html = page.content()
            browser.close()
            return BeautifulSoup(html, 'html.parser')
    except Exception as e:
        print(f"    ✘ JS render failed for {url}: {e}")
        return None

def fetch_page(url, headers, require_js=False):
    soup = get_clean_soup(url, headers)
    if soup and not require_js:
        return soup
    if playwright_available:
        js_soup = render_page_html(url, headers)
        if js_soup:
            return js_soup
    return soup

def parse_json_ld_products(soup, base_url):
    candidates = []
    for script in soup.find_all('script', type='application/ld+json'):
        try:
            payload = json.loads(script.string or script.text)
        except Exception:
            continue
        objects = payload if isinstance(payload, list) else [payload]
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            if obj.get('@type') == 'Product':
                title = obj.get('name') or obj.get('headline')
                image = obj.get('image')
                if isinstance(image, list):
                    image = image[0]
                price = None
                if isinstance(obj.get('offers'), dict):
                    price = obj['offers'].get('price') or obj['offers'].get('priceSpecification', {}).get('price')
                if title and image:
                    candidates.append({
                        'title': title.strip(),
                        'price': format_price(price),
                        'image_url': urljoin(base_url, image)
                    })
            if obj.get('@type') in ['ItemList', 'Collection'] and 'itemListElement' in obj:
                for item in obj['itemListElement']:
                    product = item.get('item') or item
                    if isinstance(product, dict) and product.get('@type') == 'Product':
                        title = product.get('name')
                        image = product.get('image')
                        if isinstance(image, list):
                            image = image[0]
                        price = None
                        if isinstance(product.get('offers'), dict):
                            price = product['offers'].get('price')
                        if title and image:
                            candidates.append({
                                'title': title.strip(),
                                'price': format_price(price),
                                'image_url': urljoin(base_url, image)
                            })
    return candidates

def extract_js_embedded_products(soup, base_url):
    candidates = []
    json_texts = []
    for script in soup.find_all('script'):
        if not script.string:
            continue
        text = script.string.strip()
        if 'window.__INITIAL_STATE__' in text or 'window.__NEXT_DATA__' in text or 'JSON.parse(' in text:
            json_texts.append(text)
    for text in json_texts:
        json_strs = re.findall(r'({.*?})', text, flags=re.S)
        for candidate_str in json_strs:
            try:
                payload = json.loads(candidate_str)
            except Exception:
                continue
            for key, value in payload.items() if isinstance(payload, dict) else []:
                if isinstance(value, dict) and value.get('name') and value.get('image'):
                    title = value.get('name')
                    image = value.get('image')
                    if isinstance(image, list):
                        image = image[0]
                    price = value.get('price') or value.get('offers', {}).get('price')
                    if title and image:
                        candidates.append({
                            'title': title.strip(),
                            'price': format_price(price),
                            'image_url': urljoin(base_url, image)
                        })
    return candidates

def extract_custom_candidates(soup, base_url):
    candidates = []
    containers = soup.find_all(['div', 'li', 'article', 'section', 'tr'])
    price_pattern = re.compile(r'(KSh|KES|₨|/=|UGX|TZS|RWF|NGN|USD|EUR|£|¥|₹|\d{2,3}(?:[.,]\d{2,3})?)')

    for container in containers:
        img = container.find('img')
        if not img:
            continue
        text = container.get_text(separator=' ', strip=True)
        if len(text) < 20 or len(text) > 800:
            continue
        price_match = price_pattern.search(text)
        if not price_match:
            continue
        src = img.get('src') or img.get('data-src') or img.get('data-lazy-src') or img.get('data-original')
        if not src:
            continue
        src_lower = src.lower()
        if any(x in src_lower for x in ['logo', 'banner', 'icon', 'sprite', 'avatar', 'facebook', 'twitter', 'instagram']):
            continue

        title = None
        title_tag = container.find(['h1', 'h2', 'h3', 'h4', 'h5', 'a', 'span'])
        if title_tag:
            title_text = title_tag.get_text(separator=' ', strip=True)
            if title_text and title_text.lower() not in ['shop', 'product', 'view details', 'add to cart', 'buy now']:
                title = title_text

        if not title:
            text_parts = [part.strip() for part in text.split('\n') if part.strip()]
            title = text_parts[0] if text_parts else None

        price_raw = price_match.group(0)
        candidates.append({
            'title': title,
            'price': format_price(price_raw),
            'image_url': urljoin(base_url, src),
            'raw_text': text
        })

    unique_candidates = []
    seen = set()
    for c in candidates:
        key = (c.get('image_url'), c.get('title'))
        if key not in seen:
            unique_candidates.append(c)
            seen.add(key)
    return unique_candidates

# ==========================================
# 3. THE HARVESTER (Core Logic)
# ==========================================
def harvester(url):
    print(f"--- [1/3] Harvesting: {url} ---")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    products = []
    custom_candidates = []
    
    try:
        shopify_url = url.rstrip('/') + '/products.json'
        res = requests.get(shopify_url, headers=headers, timeout=5)
        if res.status_code == 200:
            raw_data = res.json().get('products', [])[:10]
            for p in raw_data:
                products.append({
                    "title": p.get("title"),
                    "price": format_price(p.get("variants", [{}])[0].get("price")),
                    "image_url": p.get("images", [{}])[0].get("src") if p.get("images") else None
                })
    except:
        pass

    try:
        if not products:
            woo_url = url.rstrip('/') + '/wp-json/wc/store/products'
            res = requests.get(woo_url, headers=headers, timeout=5)
            if res.status_code == 200:
                raw_data = res.json()
                for p in raw_data[:10]:
                    products.append({
                        "title": p.get("name"),
                        "price": format_price(p.get("prices", {}).get("price")),
                        "image_url": p.get("images", [{}])[0].get("src") if p.get("images") else None
                    })
    except:
        pass

    soup = fetch_page(url, headers)
    if not soup:
        return {"products": products, "custom_candidates": [], "text": "", "url": url}

    display_soup = BeautifulSoup(str(soup), 'html.parser')
    for noise in display_soup(["script", "style", "noscript", "svg"]):
        noise.extract()
    raw_text = f"[HOME PAGE]\n{display_soup.get_text(separator=' ', strip=True)[:10000]}\n\n"

    if not products:
        custom_candidates.extend(parse_json_ld_products(soup, url))
        custom_candidates.extend(extract_js_embedded_products(soup, url))
        custom_candidates.extend(extract_custom_candidates(soup, url))

        if not custom_candidates and playwright_available:
            js_soup = render_page_html(url, headers)
            if js_soup:
                soup = js_soup
                custom_candidates.extend(parse_json_ld_products(soup, url))
                custom_candidates.extend(extract_js_embedded_products(soup, url))
                custom_candidates.extend(extract_custom_candidates(soup, url))

    important_links = {}
    target_categories = {
        'shop': ['shop', 'store', 'products', 'collection'],
        'service': ['services', 'book', 'consultation', 'solutions', 'offerings'],
        'faq': ['faq', 'questions', 'support', 'help'],
        'pricing': ['pricing', 'plans', 'packages', 'rates'],
        'about': ['about', 'story', 'who-we-are'],
        'contact': ['contact', 'reach-us', 'location']
    }

    for a in soup.find_all('a', href=True):
        link_text = a.get_text().strip().lower()
        link_href = a['href'].strip().lower()
        full_url = urljoin(url, a['href'])
        
        if urlparse(full_url).netloc != urlparse(url).netloc:
            continue
            
        for category, keywords in target_categories.items():
            if category not in important_links:
                if any(k in link_text or k in link_href for k in keywords):
                    important_links[category] = full_url

    for category, link in important_links.items():
        if link == url: 
            continue
        sub_soup = fetch_page(link, headers)
        if sub_soup:
            if category == 'shop' and not products and not custom_candidates:
                custom_candidates.extend(parse_json_ld_products(sub_soup, link))
                custom_candidates.extend(extract_custom_candidates(sub_soup, link))
                
            for noise in sub_soup(["script", "style", "svg"]):
                noise.extract()
            
            extracted_text = sub_soup.get_text(separator=' ', strip=True)[:8000]
            raw_text += f"[{category.upper()} PAGE]\n{extracted_text}\n\n"

    unique_custom = []
    seen = set()
    for c in custom_candidates:
        if c.get('image_url'):
            key = (c.get('title'), c.get('image_url'))
            if key not in seen:
                unique_custom.append(c)
                seen.add(key)

    return {
        "products": products[:20],
        "custom_candidates": unique_custom[:20],
        "text": raw_text[:35000], 
        "url": url
    }

# ==========================================
# 4. THE CHEF (AI Cleaning logic)
# ==========================================
def chef(harvested_data):
    system_prompt = """
    You are an elite Data Architect extracting context for an autonomous sales AI.
    First, determine the 'business_type'. It MUST be either "ecommerce" or "service".
    
    If "ecommerce", your JSON output MUST include:
    - business_name, business_type ("ecommerce"), brand_voice, unique_selling_point
    - shipping_policy, return_policy, payment_methods
    - top_10_products: list of dicts with 'title', 'price', 'image_url'
    
    If "service", your JSON output MUST include:
    - business_name, business_type ("service"), brand_voice, unique_selling_point
    - service_packages: list of dicts with 'service_name', 'description', 'price' (if available)
    - booking_process: how does a client start? (e.g., "book a call", "fill a form")
    - service_areas: locations covered
    
    FOR BOTH TYPES, YOU MUST INCLUDE:
    - contact_info: object with phone, email, physical_location (if found)
    - faq_and_objections: list of dicts with 'question' and 'answer' extracted from the text.
    - social_proof: list of quotes or stats showing past success.
    
    CRITICAL: 
    1. If prices in 'custom_candidates' look like '699900', format them as '6,999.00'.
    2. Ignore UI elements (e.g., 'Cart', 'Login', 'Menu').
    """

    user_content = f"WEBSITE TEXT: {harvested_data['text']}\n\n"
    if harvested_data.get('products'):
        user_content += f"STRUCTURED DATA: {harvested_data['products']}"
    else:
        user_content += f"RAW HTML CANDIDATES: {harvested_data.get('custom_candidates', [])}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={ "type": "json_object" }
        )
        
        profile = json.loads(response.choices[0].message.content)
        
        if profile.get('business_type') == 'ecommerce' and harvested_data.get('products'):
            profile['top_10_products'] = harvested_data['products']
            
        return profile
    except Exception as e:
        return None
    
# ==========================================
# 5. THE ARCHITECT (Strategic Prompt Engineer)
# ==========================================
def architect(profile, raw_text):
    system_prompt = """
    You are an expert Prompt Engineer for Social Commerce AI. 
    Your task is to transform a business JSON Profile and Raw Scraped Text into a neutral, highly instructive 'Business Essence' file.
    
    This file will be used as a mid-layer instruction set for an AI agent. 
    Avoid technical AI jargon. Focus entirely on the business logic and facts.
    
    YOUR OUTPUT MUST FOLLOW THIS STRUCTURE:

    1. BUSINESS IDENTITY & ESSENCE
    - Core Role: Define the professional persona (e.g., 'A high-end bridal consultant').
    - Brand Voice: Specific linguistic traits (e.g., 'Professional but warm, using Kenyan hospitality nuances').
    - The Competitive Edge: Detailed list of USPs extracted from the raw text.

    2. LOGIC-DRIVEN WORKFLOW (If-Then Sales Funnel)
    Define the interaction flow using 'If the user does X, the system must do Y' logic.
    - PHASE 1: Hook & Qualify -> How to greet and the specific question needed to identify if they are a lead.
    - PHASE 2: Discovery -> The exact data points the system must extract before making a recommendation.
    - PHASE 3: The Pitch -> How to present the solution based on the discovery. 
    - PHASE 4: Objection Handling -> Specific logic for rebutting common concerns (Price, Trust, Timelines) using business-specific facts.
    - PHASE 5: The Close -> The exact criteria for a successful close and the specific call-to-action (WhatsApp, Link, etc.).

    3. BUSINESS GUARDRAILS
    - Strict Prohibitions: List specific things the business never does or says (e.g., 'Never promise delivery in under 24 hours').
    - Handover Triggers: Specific scenarios where the system must stop and request human intervention.

    4. PRODUCT/SERVICE CATALOGUE
    - Organized categories, detailed descriptions, and pricing extracted from the text.
    - Logistics & Operations: Shipping, returns, booking windows, and payment expectations (e.g., M-PESA context).

    5. OFFICIAL CONTACTS & TRUST SIGNALS
    - List all physical addresses, phone numbers, emails, and social proof found.

    CRITICAL REQUIREMENTS:
    - Instructivity: Use 'If/Then' logic and 'System Must' directives. 
    - No 'But-Then' storytelling: Focus on direct, logic-based positioning.
    - Authenticity: Reference specific details from the 'Raw Text' (specific locations, local phrases, or unique policies).
    - Neutrality: Do not refer to 'AI', 'Prompts', or 'Chatbots'. Treat this as a manual for a perfect employee.
    """

    user_content = f"JSON PROFILE: {json.dumps(profile)}\n\nRAW SCRAPED TEXT: {raw_text[:20000]}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return None

# ==========================================
# 6. FASTAPI APPLICATION
# ==========================================
app = FastAPI()

class OnboardingRequest(BaseModel):
    url: str

@app.post("/generate-profile")
def generate_business_profile(req: OnboardingRequest):
    try:
        raw_info = harvester(req.url)
        
        final_profile = chef(raw_info)
        if not final_profile:
            raise HTTPException(status_code=500, detail="Chef failed to generate profile.")
            
        bot_instructions = architect(final_profile, raw_info['text'])
        if not bot_instructions:
            raise HTTPException(status_code=500, detail="Architect failed to generate instructions.")
            
        business_name = final_profile.get("business_name", "New Business")
        
        return {
            "name": business_name,
            "profile_json": final_profile,
            "instructions_md": bot_instructions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Scraper is active"}

@app.get("/kaithhealthcheck")
@app.get("/kaithheathcheck")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
