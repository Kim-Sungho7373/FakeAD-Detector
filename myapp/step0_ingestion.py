import os
import gc
import requests
import torch
import whisper
import yt_dlp
from paddleocr import PaddleOCR
from playwright.sync_api import sync_playwright
import logging

class DataIngestionPipeline:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"✅ 사용 중인 디바이스: {self.device}")

    def clear_memory(self):
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print("🧹 메모리 정리 완료")

    def extract_audio_from_video(self, video_url, output_filename="temp_audio"):
        print(f"\n🎥 [1] 유튜브 영상 다운로드 시작 (yt-dlp): {video_url}")
        
        # 유튜브 로봇 차단 우회 및 mp3 추출 최적화 옵션
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{output_filename}.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # 변환된 mp3 파일명
            final_path = f"{output_filename}.mp3"
            print(f"✅ 오디오 추출 완료: {final_path}")
            return final_path
        except Exception as e:
            print(f"❌ 다운로드 실패: {e}")
            return None

    def run_stt(self, audio_path):
        print(f"\n🗣️ [2] STT(음성->텍스트) 변환 시작: {audio_path}")
        model = whisper.load_model("small", device="cpu")
        result = model.transcribe(audio_path, language="ko", fp16=False)
        
        if result is None:
            raise ValueError("Whisper가 텍스트를 반환하지 못했습니다.")
            
        text_result = result.get("text", "")
        
        del model
        self.clear_memory()
        
        print("✅ STT 변환 완료")
        return text_result.strip()

    def run_ocr_from_web(self, product_url):
        print(f"\n🖼️ [3] 웹페이지 접속 및 이미지 OCR 시작: {product_url}")
        
        raw_image_urls = []
        with sync_playwright() as p:
            print("   -> 브라우저 창을 백그라운드로 띄우고 페이지 로딩을 기다립니다...")
            # 🚨 중요: 허깅페이스(서버) 환경을 위한 headless=True 및 도커 필수 옵션 추가!
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
                
                # 🌟 방금 확인한 완벽한 데이터 추출 로직 적용!
                if result and isinstance(result[0], dict) and 'rec_texts' in result[0]:
                    texts = result[0]['rec_texts']
                    print(f"      => ✨ 텍스트 {len(texts)}줄 추출 성공!")
                    all_extracted_text.extend(texts)
                    if os.path.exists(temp_img_path): os.remove(temp_img_path)
                else:
                    print(f"      => ⚠️ 글자가 없습니다! (이미지 확인: {temp_img_path})")
                                
            except Exception as e:
                print(f"⚠️ 이미지 처리 중 오류 발생: {e}")
                if os.path.exists(temp_img_path): os.remove(temp_img_path)
                    
        del ocr
        self.clear_memory()
        
        final_text = "\n".join(all_extracted_text) # 줄바꿈으로 깔끔하게 합치기
        print("\n✅ 웹페이지 이미지 OCR 변환 완료!")
        return final_text
    