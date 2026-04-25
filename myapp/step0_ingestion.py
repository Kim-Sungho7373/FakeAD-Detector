import os
import gc
import requests
from paddleocr import PaddleOCR
from playwright.sync_api import sync_playwright
import logging

class DataIngestionPipeline:
    def __init__(self):
        # Set logging to suppress unnecessary info
        logging.getLogger('ppocr').setLevel(logging.ERROR)
        print("✅ DataIngestionPipeline initialized (English/Global Context)")

    def clear_memory(self):
        gc.collect()
        print("🧹 Memory cleanup complete")

    def run_ocr_from_web(self, product_url):
        print(f"\n🌐 [Step 0] Auditing Web Content: {product_url}")
        
        extracted_html_text = ""
        raw_image_urls = []
        
        with sync_playwright() as p:
            print("   -> Launching headless browser...")
            browser = p.chromium.launch(
                headless=True, 
                args=["--no-sandbox", "--disable-setuid-sandbox"]
            ) 
            page = browser.new_page()
            
            # Increase timeout for global sites
            page.goto(product_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000) 
            
            # [Core Logic 1] Extracting Visible HTML Text (Crucial for Dark Patterns)
            # Dark patterns are often in the footer or tiny text labels.
            print("   -> 📄 Extracting visible text from HTML (Footer/Terms/Body)...")
            extracted_html_text = page.evaluate("() => document.body.innerText")

            # [Core Logic 2] Handling English "See More" Buttons
            print("   -> 🎯 Looking for disclosure buttons (Details/Terms/More)...")
            try:
                # Updated selectors for English subscription sites
                selectors = [
                    'button:has-text("Show more")', 
                    'button:has-text("See details")', 
                    'button:has-text("Terms")', 
                    'a:has-text("Read more")',
                    'span:has-text("View more")'
                ]
                for selector in selectors:
                    btn = page.locator(selector).first
                    if btn.is_visible(timeout=2000):
                        btn.click()
                        print(f"      => Clicked: {selector}")
                        page.wait_for_timeout(1000)
            except Exception:
                pass

            # Scroll to trigger lazy loading for images
            print("   -> Scrolling to load billing disclosures...")
            for _ in range(5):
                page.evaluate("window.scrollBy(0, 1000)")
                page.wait_for_timeout(800) 
            
            # Collect Image URLs for OCR
            img_elements = page.query_selector_all('img')
            for img in img_elements:
                src = img.get_attribute('data-src') or img.get_attribute('src')
                if src and ('http' in src or src.startswith('//')):
                    if src.startswith('//'): src = 'https:' + src
                    raw_image_urls.append(src)
            
            browser.close()

        # Image Filtering
        valid_urls = [url for url in raw_image_urls if not any(x in url.lower() for x in ['.gif', 'icon', 'logo', 'svg', 'thumb'])]

        # [Core Logic 3] OCR for English Text
        all_extracted_text = [extracted_html_text] # Start with HTML text
        
        if valid_urls:
            print(f"✅ Found {len(valid_urls)} images. Running English OCR on key visual assets...")
            
            # Change language to 'en' for English detection
            ocr = PaddleOCR(
                use_angle_cls=True, 
                lang='en', # CRITICAL: Changed to English
                show_log=False
            )
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }

            processed_count = 0
            # Scan top 3 major images (likely billing banners)
            for i, img_url in enumerate(valid_urls[:5]): 
                if processed_count >= 3: break
                    
                temp_img_path = f"temp_ocr_{i}.jpg"
                try:
                    response = requests.get(img_url, headers=headers, timeout=10)
                    with open(temp_img_path, 'wb') as f:
                        f.write(response.content)
                    
                    if os.path.getsize(temp_img_path) < 40000: # Ignore small icons
                        if os.path.exists(temp_img_path): os.remove(temp_img_path)
                        continue
                        
                    processed_count += 1
                    result = ocr.ocr(temp_img_path, cls=True)
                    
                    if result and result[0]:
                        texts = [line[1][0] for line in result[0]]
                        print(f"      => ✨ Extracted {len(texts)} lines from image {i}.")
                        all_extracted_text.extend(texts)
                                    
                except Exception as e:
                    print(f"⚠️ Image processing error: {e}")
                finally:
                    if os.path.exists(temp_img_path): os.remove(temp_img_path)
            
            del ocr

        self.clear_memory()
        
        # Merge all text (HTML Text + OCR Text)
        final_text = "\n".join(all_extracted_text).strip()
        print(f"✅ Web Audit Complete. Total characters extracted: {len(final_text)}")
        return final_text
    