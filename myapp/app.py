import os
import math
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================================
# 🌟 AI 레고 블록 불러오기
# ==========================================
# 폴더 구조에 따라 import 경로가 달라질 수 있습니다. 
# myapp 폴더 안에서 실행된다면 아래와 같이 유지하세요.
from step0_ingestion import DataIngestionPipeline
from step1_lexical import LexicalAnalyzer
from step2_semantic import SemanticAnalyzer
from step3_rag import FactCheckerRAG
from step4_xai import XAIScorer

app = FastAPI()

# 🌟 외부 호출 허용 (CORS 설정)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("==================================================")
print(" ⏳ AI 엔진 및 딥러닝 모델들을 메모리에 올리는 중입니다...")
print("==================================================")
# Render 무료 플랜(512MB)에서 메모리 부족이 발생할 경우, 
# 이 초기화 부분들을 함수 내부로 옮겨야 할 수도 있습니다.
ingestion = DataIngestionPipeline()
lexical = LexicalAnalyzer()
semantic = SemanticAnalyzer()
rag_checker = FactCheckerRAG()
xai_scorer = XAIScorer()
print("\n✅ [서버 준비 완료]\n")

class AdRequest(BaseModel):
    video_url: str
    product_url: str

# 1. 🌟 대시보드 웹 프론트엔드 (HTML)
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    # 제공해주신 HTML 코드를 그대로 유지합니다.
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <title>AI 과대광고 탐지 대시보드</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: 'Pretendard', sans-serif; background-color: #121212; color: #ffffff; padding: 20px; margin: 0; }
            .container { max-width: 1200px; margin: auto; }
            h2 { color: #ffffff; text-align: left; border-bottom: 2px solid #333; padding-bottom: 10px; }
            .input-section { background: #1e1e1e; padding: 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            input { width: 100%; padding: 12px; margin: 8px 0; background: #2c2c2e; border: 1px solid #444; border-radius: 8px; color: #fff; box-sizing: border-box; }
            button { width: 100%; padding: 15px; background: #4caf50; color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; transition: 0.3s; }
            .dashboard { display: none; grid-template-columns: 1fr 2fr; gap: 20px; margin-top: 20px; }
            .card { background: #1e1e1e; padding: 20px; border-radius: 12px; }
            #loading { display: none; text-align: center; color: #4caf50; font-size: 18px; margin-top: 20px; font-weight: bold; }
            .danger { color: #ff5252; font-weight: bold; }
            .highlight { color: #4caf50; font-weight: bold; }
            .details-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
            .detail-item { background: #2c2c2e; padding: 15px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🚨 AI 과대광고 탐지 대시보드 (Overview)</h2>
            <div class="input-section">
                <input type="text" id="video_url" placeholder="유튜브 영상 링크 (선택사항)">
                <input type="text" id="product_url" placeholder="상품 상세페이지 링크 (필수)">
                <button onclick="analyzeAd()">분석 시작</button>
            </div>
            <div id="loading">분석 중입니다...</div>
            <div class="dashboard" id="dashboard">
                <div class="card"><h3>위험도</h3><canvas id="scoreChart"></canvas><h1 id="scoreText"></h1><p id="statusText"></p></div>
                <div class="card"><h3>AI 사고회로</h3><div id="xaiReasoning"></div></div>
            </div>
            <div class="dashboard" id="detailsDashboard" style="grid-template-columns: 1fr;">
                <div class="card">
                    <h3>세부 분석 결과</h3>
                    <div class="details-grid">
                        <div id="x1Details" class="detail-item"></div>
                        <div id="x2Details" class="detail-item"></div>
                        <div id="x3Details" class="detail-item" style="grid-column: span 2;"></div>
                    </div>
                </div>
            </div>
        </div>
        <script>
            async function analyzeAd() {
                const videoUrl = document.getElementById('video_url').value;
                const productUrl = document.getElementById('product_url').value;
                document.getElementById('loading').style.display = 'block';
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ video_url: videoUrl, product_url: productUrl })
                });
                const data = await response.json();
                document.getElementById('loading').style.display = 'none';
                if (data.status === "success") {
                    document.getElementById('dashboard').style.display = 'grid';
                    document.getElementById('detailsDashboard').style.display = 'grid';
                    document.getElementById('scoreText').innerText = data.final_score + "점";
                    document.getElementById('xaiReasoning').innerHTML = data.xai_reasoning;
                    document.getElementById('x1Details').innerHTML = data.x1_details;
                    document.getElementById('x2Details').innerHTML = data.x2_details;
                    document.getElementById('x3Details').innerHTML = data.x3_details;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# 2. 분석 API
@app.post("/api/analyze")
async def api_analyze(req: AdRequest):
    try:
        # Step 0: 데이터 수집
        stt_text = ingestion.run_stt(ingestion.extract_audio_from_video(req.video_url)) if req.video_url.strip() else ""
        ocr_text = ingestion.run_ocr_from_web(req.product_url) if req.product_url.strip() else ""
        combined_text = f"{stt_text}\n{ocr_text}".strip()
        
        if len(combined_text) < 5:
            return {"status": "error", "error": "텍스트를 찾지 못했습니다."}

        x1_score = lexical.calculate_x1_score(combined_text)
        x2_score = semantic.calculate_x2_score(combined_text)
        x3_score, matched_fact = rag_checker.calculate_x3_score(combined_text)
        
        final_score, shap_vals, _ = xai_scorer.calculate_final_score_and_explain(x1_score, x2_score, x3_score)
        
        # UI용 텍스트 생성 (제공해주신 로직과 동일)
        detected_words = [word for word in lexical.lexicon.keys() if word in combined_text]
        x1_details = f"적발 단어: {', '.join(detected_words)}" if detected_words else "금칙어 없음"
        
        x2_details = f"문맥 위험도: {x2_score:.1f}점"
        x3_details = f"참조 규정: {matched_fact}"

        xai_reasoning = "AI 분석 완료" # 간단하게 축약

        return {
            "status": "success",
            "final_score": float(round(final_score, 2)),
            "x1_details": x1_details,
            "x2_details": x2_details,
            "x3_details": x3_details,
            "xai_reasoning": xai_reasoning
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    # 로컬 테스트용
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)