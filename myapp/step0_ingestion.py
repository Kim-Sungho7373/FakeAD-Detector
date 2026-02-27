import os
import gc
import requests
from paddleocr import PaddleOCR
from playwright.sync_api import sync_playwright
import logging

class DataIngestionPipeline:
    def __init__(self):
        # STT, MPS(Mac GPU) 관련 설정 제거 및 간소화
        print("✅ DataIngestionPipeline 초기화 완료 (웹 크롤링 및 OCR 전용)")

    def clear_memory(self):
        gc.collect()
        print("🧹 메모리 정리 완료")

    def run_ocr_from_web(self, product_url):
        print(f"\n🖼️ [1] 웹페이지 접속 및 이미지 OCR 시작: {product_url}")
        
        raw_image_urls = []
        with sync_playwright() as p:
            print("   -> 브라우저 창을 백그라운드로 띄우고 페이지 로딩을 기다립니다...")
            # 🚨 허깅페이스(서버) 환경을 위한 headless=True 및 도커 필수 옵션 유지
            browser = p.chromium.launch(
                headless=True, 
                args=["--no-sandbox", "--disable-setuid-sandbox"]
            ) 
            page = browser.new_page()
            page.goto(product_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(3000) 
            
            print("   -> 🎯 숨겨진 상세페이지를 열기 위해 '더보기' 버튼을 찾습니다...")
            try:
                more_btn = page.locator('button:has-text("상세정보 펼쳐보기"), button:has-text("상세설명 더보기"), a:has-text("더보기")').first
                if more_btn.is_visible(timeout=3000):
                    more_btn.click()
                    print("      => 쾅! '더보기' 버튼을 성공적으로 클릭했습니다!")
                    page.wait_for_timeout(2000) 
            except Exception:
                print("      => '더보기' 버튼이 없거나 이미 펼쳐져 있습니다. 그대로 진행합니다.")

            print("   -> 지연 로딩(Lazy-loading)된 이미지를 불러오기 위해 스크롤을 내립니다...")
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
            print("❌ 유효한 상세 이미지를 찾을 수 없습니다.")
            return ""

        print(f"✅ 총 {len(valid_urls)}개의 이미지 발견! 진짜 상세 이미지를 탐색합니다...")
        
        logging.getLogger('ppocr').setLevel(logging.ERROR)
        
        # 🌟 해상도 한계치를 대폭 늘린 최신 세팅 적용
        ocr = PaddleOCR(
            lang='korean',
            use_gpu=False, # 허깅페이스 CPU 환경 명시
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
                print(f"   -> [진짜 텍스트 탐색 중...] 묵직한 상세 이미지 발견! ({file_kb}KB)")
                
                result = ocr.ocr(temp_img_path) 
                
                # 🌟 기존에 검증된 데이터 추출 로직 유지
                if result and isinstance(result[0], dict) and 'rec_texts' in result[0]:
                    texts = result[0]['rec_texts']
                    print(f"      => ✨ 텍스트 {len(texts)}줄 추출 성공!")
                    all_extracted_text.extend(texts)
                elif result and isinstance(result[0], list):
                    # 일반적인 PaddleOCR 반환 형식 대응 로직 추가 (안전장치)
                    texts = [line[1][0] for line in result[0] if line is not None]
                    print(f"      => ✨ 텍스트 {len(texts)}줄 추출 성공!")
                    all_extracted_text.extend(texts)
                else:
                    print(f"      => ⚠️ 글자가 없습니다! (이미지 확인: {temp_img_path})")
                                
            except Exception as e:
                print(f"⚠️ 이미지 처리 중 오류 발생: {e}")
            finally:
                # 성공/실패 여부와 상관없이 임시 이미지 파일 삭제 보장
                if os.path.exists(temp_img_path): 
                    os.remove(temp_img_path)
                    
        del ocr
        self.clear_memory()
        
        final_text = "\n".join(all_extracted_text).strip()
        print("\n✅ 웹페이지 이미지 OCR 변환 완료!")
        return final_text
    