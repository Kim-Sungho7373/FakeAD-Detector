import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================================
# 🌟 AI 레고 블록 불러오기
# ==========================================
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
print(" ⏳ [Hugging Face Space] AI 엔진 모델 로딩 중...")
print("==================================================")
# HF Spaces는 RAM이 16GB로 넉넉하므로 글로벌로 로딩해두면
# 매 요청마다 모델을 새로 부르지 않아 응답 속도가 훨씬 빠릅니다.
ingestion = DataIngestionPipeline()
lexical = LexicalAnalyzer()
semantic = SemanticAnalyzer()
rag_checker = FactCheckerRAG()
xai_scorer = XAIScorer()
print("\n✅ [서버 준비 완료]\n")

# 🌟 1. 데이터 모델 수정: video_url 완전 제거, product_url만 받음
class AdRequest(BaseModel):
    product_url: str

# 🌟 2. 대시보드 웹 프론트엔드 (HTML)
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
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
            button:disabled { background: #555; cursor: not-allowed; opacity: 0.7; }
            .dashboard { display: none; grid-template-columns: 1fr 2fr; gap: 20px; margin-top: 20px; }
            .card { background: #1e1e1e; padding: 20px; border-radius: 12px; }
            
            #loading-spinner { display: none; text-align: center; margin: 30px 0; }
            .spinner { display: inline-block; width: 50px; height: 50px; border: 5px solid #333; border-top: 5px solid #4caf50; border-radius: 50%; animation: spin 1s linear infinite; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .loading-text { margin-top: 15px; font-size: 16px; font-weight: bold; color: #4caf50; }
            .loading-subtext { font-size: 14px; color: #888; font-weight: normal; }

            .details-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
            .detail-item { background: #2c2c2e; padding: 15px; border-radius: 8px; line-height: 1.5; }
            #error-message { display: none; background: #ff5252; color: white; padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🚨 AI 과대광고 탐지 대시보드</h2>
            <div class="input-section">
                <input type="text" id="product_url" placeholder="분석할 상품 상세페이지 또는 웹페이지 링크를 입력하세요 (필수)">
                <button id="analyze-btn" onclick="analyzeAd()">크롤링 및 AI 분석 시작</button>
            </div>
            
            <div id="loading-spinner">
                <div class="spinner"></div>
                <p class="loading-text">
                    🧠 AI가 웹페이지 텍스트를 추출하고 팩트체크를 진행 중입니다...<br>
                    <span class="loading-subtext">(잠시만 기다려주세요)</span>
                </p>
            </div>

            <div id="error-message"></div>

            <div class="dashboard" id="dashboard">
                <div class="card">
                    <h3>최종 위험 점수</h3>
                    <h1 id="scoreText" style="font-size: 48px; margin: 10px 0; color: #ff5252;"></h1>
                </div>
                <div class="card">
                    <h3>AI 사고회로 (설명 가능 인공지능)</h3>
                    <div id="xaiReasoning"></div>
                </div>
            </div>
            <div class="dashboard" id="detailsDashboard" style="grid-template-columns: 1fr;">
                <div class="card">
                    <h3>세부 분석 결과</h3>
                    <div class="details-grid">
                        <div class="detail-item"><strong>1. 금칙어 탐지:</strong><br><span id="x1Details"></span></div>
                        <div class="detail-item"><strong>2. 문맥 위험도:</strong><br><span id="x2Details"></span></div>
                        <div class="detail-item" style="grid-column: span 2;"><strong>3. 규정/팩트체크 위반 사항:</strong><br><span id="x3Details"></span></div>
                    </div>
                </div>
            </div>
        </div>
        <script>
            async function analyzeAd() {
                const productUrl = document.getElementById('product_url').value;
                
                if (!productUrl.trim()) {
                    alert("웹페이지 링크를 입력해주세요.");
                    return;
                }

                const loadingSpinner = document.getElementById('loading-spinner');
                const analyzeBtn = document.getElementById('analyze-btn');
                const dashboard = document.getElementById('dashboard');
                const detailsDashboard = document.getElementById('detailsDashboard');
                const errorMsg = document.getElementById('error-message');

                loadingSpinner.style.display = 'block';
                analyzeBtn.disabled = true;
                analyzeBtn.innerText = '분석 중... ⏳';
                dashboard.style.display = 'none';
                detailsDashboard.style.display = 'none';
                errorMsg.style.display = 'none';

                try {
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ product_url: productUrl }) // 데이터 전송 형식 변경
                    });
                    
                    const data = await response.json();
                    
                    if (data.status === "success") {
                        dashboard.style.display = 'grid';
                        detailsDashboard.style.display = 'grid';
                        document.getElementById('scoreText').innerText = data.final_score + "점";
                        document.getElementById('xaiReasoning').innerHTML = data.xai_reasoning || "AI 분석 완료";
                        document.getElementById('x1Details').innerHTML = data.x1_details || "적발 내역 없음";
                        document.getElementById('x2Details').innerHTML = data.x2_details || "안전함";
                        document.getElementById('x3Details').innerHTML = data.x3_details || "규정 위반 없음";
                    } else {
                        errorMsg.innerText = "❌ 분석 실패: " + data.error;
                        errorMsg.style.display = 'block';
                    }
                } catch (error) {
                    console.error('Fetch 에러:', error);
                    errorMsg.innerText = "❌ 서버와 통신하는 중 문제가 발생했습니다.";
                    errorMsg.style.display = 'block';
                } finally {
                    loadingSpinner.style.display = 'none';
                    analyzeBtn.disabled = false;
                    analyzeBtn.innerText = '크롤링 및 AI 분석 시작';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# 🌟 3. 분석 API (웹 크롤링만 수행)
@app.post("/api/analyze")
def api_analyze(req: AdRequest):
    try:
        # Step 0: 데이터 수집 (웹 크롤링만 진행)
        # ingestion.run_ocr_from_web() 메서드가 BeautifulSoup 등을 이용해 텍스트를 추출한다고 가정
        crawled_text = ingestion.run_ocr_from_web(req.product_url) if req.product_url.strip() else ""
        
        if len(crawled_text) < 5:
            return {"status": "error", "error": "웹페이지에서 충분한 텍스트를 추출하지 못했습니다. 크롤링 방지가 적용된 사이트일 수 있습니다."}

        # 분석 파이프라인 진행
        x1_score = lexical.calculate_x1_score(crawled_text)
        x2_score = semantic.calculate_x2_score(crawled_text)
        x3_score, matched_fact = rag_checker.calculate_x3_score(crawled_text)
        
        final_score, shap_vals, _ = xai_scorer.calculate_final_score_and_explain(x1_score, x2_score, x3_score)
        
        # UI용 텍스트 생성
        detected_words = [word for word in lexical.lexicon.keys() if word in crawled_text]
        x1_details = f"적발 단어: <span style='color:#ff5252;'>{', '.join(detected_words)}</span>" if detected_words else "금칙어 없음"
        
        x2_details = f"문맥 위험도 점수: {x2_score:.1f}점"
        x3_details = f"참조 규정: {matched_fact}"
        xai_reasoning = "AI가 문맥과 규정을 종합하여 분석한 결과입니다." 

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
    # Hugging Face Spaces는 기본적으로 7860 포트를 외부로 노출합니다.
    # host를 0.0.0.0으로 해야 외부에서 접근 가능합니다.
    uvicorn.run("app:app", host="0.0.0.0", port=7860)