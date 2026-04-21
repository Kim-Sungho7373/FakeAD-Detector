import os
import gc
import requests
from paddleocr import PaddleOCR
from playwright.sync_api import sync_playwright
import logging

class DataIngestionPipeline:
    def __init__(self):
        print("✅ DataIngestionPipeline initialized (web crawling + OCR only)")

    def clear_memory(self):
        gc.collect()
        print("🧹 Memory cleanup complete")

    def run_ocr_from_web(self, product_url):
        print(f"\n🖼️ [1] Starting webpage access and image OCR: {product_url}")
        
        raw_image_urls = []
        with sync_playwright() as p:
            print("   -> Launching a background browser and waiting for the page to load...")
            browser = p.chromium.launch(
                headless=True, 
                args=["--no-sandbox", "--disable-setuid-sandbox"]
            ) 
            page = browser.new_page()
            page.goto(product_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000) 
            
            print("   -> 🎯 Looking for a 'more details' button to expand hidden sections...")
            try:
                more_btn = page.locator('button:has-text("상세정보 펼쳐보기"), button:has-text("상세설명 더보기"), a:has-text("더보기")').first
                if more_btn.is_visible(timeout=3000):
                    more_btn.click()
                    print("      => Successfully clicked the expand button.")
                    page.wait_for_timeout(2000) 
            except Exception:
                print("      => No expand button was found, or the section was already open. Continuing as-is.")

            print("   -> Scrolling down to load lazy-loaded images...")
            for _ in range(10):
                page.evaluate("window.scrollBy(0, 1500)")
                page.wait_for_timeout(1000) 
            
            img_elements = page.query_selector_all('img')
            for img in img_elements:
                src = img.get_attribute('data-src') or img.get_attribute('src')
                if src and ('http' in src or src.startswith('//')):
                    if src.startswith('//'):
                        src = 'https:' + src
                    raw_image_urls.append(src)
            browser.close()

        valid_urls = []
        for url in raw_image_urls:
            url_lower = url.lower()
            if not any(x in url_lower for x in ['.gif', 'icon', 'logo', 'blank', 'svg', 'thumb']):
                valid_urls.append(url)

        if not valid_urls:
            print("❌ No valid product detail images were found.")
            return ""

        print(f"✅ Found {len(valid_urls)} candidate images. Searching for text-rich detail images...")
        
        logging.getLogger('ppocr').setLevel(logging.ERROR)
        
        ocr = PaddleOCR(
            lang='korean',
            text_det_limit_side_len=2048,
            text_det_limit_type='max'
        )
        
        all_extracted_text = []

        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': product_url
        }

        processed_count = 0
        
        for i, img_url in enumerate(valid_urls[2:]): 
            if processed_count >= 3: 
                break
                
            temp_img_path = f"temp_ocr_{i}.jpg"
            try:
                response = requests.get(img_url, headers=headers, timeout=10)
                with open(temp_img_path, 'wb') as f:
                    f.write(response.content)
                
                if os.path.getsize(temp_img_path) < 30000:
                    if os.path.exists(temp_img_path): os.remove(temp_img_path)
                    continue
                    
                processed_count += 1
                file_kb = os.path.getsize(temp_img_path) // 1024
                print(f"   -> [Scanning for real text...] Found a substantial detail image ({file_kb}KB)")
                
                result = ocr.ocr(temp_img_path) 
                
                if result and isinstance(result[0], dict) and 'rec_texts' in result[0]:
                    texts = result[0]['rec_texts']
                    print(f"      => ✨ Successfully extracted {len(texts)} lines of text.")
                    all_extracted_text.extend(texts)
                elif result and isinstance(result[0], list):
                    texts = [line[1][0] for line in result[0] if line is not None]
                    print(f"      => ✨ Successfully extracted {len(texts)} lines of text.")
                    all_extracted_text.extend(texts)
                else:
                    print(f"      => ⚠️ No text detected in this image. (Checked: {temp_img_path})")
                                
            except Exception as e:
                print(f"⚠️ Error while processing an image: {e}")
            finally:
                if os.path.exists(temp_img_path): 
                    os.remove(temp_img_path)
                    
        del ocr
        self.clear_memory()
        
        final_text = "\n".join(all_extracted_text).strip()
        print("\n✅ Webpage image OCR completed.")
        return final_text
    
