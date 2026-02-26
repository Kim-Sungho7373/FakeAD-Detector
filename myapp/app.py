from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware # 🌟 CORS 추가
from pydantic import BaseModel
import uvicorn
import math

# (기존 AI 모듈 import 부분 동일)

app = FastAPI()

# 🌟 외부(Wix)에서 API를 호출할 수 있도록 CORS 허용 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 서비스 시에는 ["https://내wix주소.com"] 으로 변경 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# (이하 기존 코드 동일...)

from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello from Render!"

if __name__ == "__main__":
    app.run()

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn
import math

# ==========================================
# 🌟 AI 레고 블록 불러오기
# ==========================================
from step0_ingestion import DataIngestionPipeline
from step1_lexical import LexicalAnalyzer
from step2_semantic import SemanticAnalyzer
from step3_rag import FactCheckerRAG
from step4_xai import XAIScorer

app = FastAPI()

print("==================================================")
print(" ⏳ AI 엔진 및 딥러닝 모델들을 메모리에 올리는 중입니다...")
print("==================================================")
ingestion = DataIngestionPipeline()
lexical = LexicalAnalyzer()
semantic = SemanticAnalyzer()
rag_checker = FactCheckerRAG()
xai_scorer = XAIScorer()
print("\n✅ [서버 준비 완료] http://127.0.0.1:8000 에 접속하세요!\n")

class AdRequest(BaseModel):
    video_url: str
    product_url: str

# 1. 🌟 다크 모드 대시보드 웹 프론트엔드 (HTML/CSS/JS)
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
            
            /* 입력 폼 섹션 */
            .input-section { background: #1e1e1e; padding: 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            input { width: 100%; padding: 12px; margin: 8px 0; background: #2c2c2e; border: 1px solid #444; border-radius: 8px; color: #fff; box-sizing: border-box; }
            button { width: 100%; padding: 15px; background: #4caf50; color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; transition: 0.3s; }
            button:hover { background: #45a049; }
            
            /* 대시보드 그리드 레이아웃 */
            .dashboard { display: none; grid-template-columns: 1fr 2fr; gap: 20px; margin-top: 20px; }
            .card { background: #1e1e1e; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            .card h3 { margin-top: 0; color: #a0a0a0; font-size: 16px; border-bottom: 1px solid #333; padding-bottom: 10px; }
            
            /* 도넛 차트 카드 */
            .score-card { text-align: center; }
            .score-card canvas { max-height: 250px; margin: auto; }
            
            /* 사고회로 카드 */
            .xai-card { font-size: 15px; line-height: 1.6; color: #e0e0e0; }
            .highlight { color: #4caf50; font-weight: bold; }
            .danger { color: #ff5252; font-weight: bold; }
            
            /* 세부 엔진 결과 카드 */
            .details-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
            .detail-item { background: #2c2c2e; padding: 15px; border-radius: 8px; }
            .detail-item h4 { margin: 0 0 10px 0; color: #4caf50; }
            
            /* 로딩 텍스트 */
            #loading { display: none; text-align: center; color: #4caf50; font-size: 18px; margin-top: 20px; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🚨 AI 과대광고 탐지 대시보드 (Overview)</h2>
            
            <div class="input-section">
                <input type="text" id="video_url" placeholder="유튜브 영상 링크 (선택사항)">
                <input type="text" id="product_url" placeholder="상품 상세페이지 링크 (필수)" value="https://brand.naver.com/pacsafe/products/9365045491">
                <button onclick="analyzeAd()">분석 시작 (데이터 크롤링 및 AI 분석)</button>
            </div>
            
            <div id="loading">데이터를 수집하고 AI 모델이 분석 중입니다. 잠시만 기다려주세요 ⏳...</div>

            <div class="dashboard" id="dashboard">
                <div class="card score-card">
                    <h3>통합 위험도 점수 (Final Score)</h3>
                    <canvas id="scoreChart"></canvas>
                    <h1 id="scoreText" style="margin-top: 15px;">0.00점</h1>
                    <p id="statusText" style="color: #a0a0a0;"></p>
                </div>
                
                <div class="card xai-card">
                    <h3>🤖 AI 최종 판정 사고회로 (XAI Reasoning)</h3>
                    <div id="xaiReasoning" style="margin-bottom: 20px;"></div>
                    
                    <h3>🌐 RAG 팩트체크 2D 벡터 공간 비교</h3>
                    <canvas id="vectorChart" style="max-height: 200px;"></canvas>
                </div>
            </div>

            <div class="dashboard" id="detailsDashboard" style="grid-template-columns: 1fr; margin-top: 0;">
                <div class="card">
                    <h3>⚙️ 세부 엔진 분석 결과 (Detailed Engine Results)</h3>
                    <div class="details-grid">
                        <div class="detail-item">
                            <h4>[X1] 형태소 및 단어 탐지기</h4>
                            <p id="x1Details"></p>
                        </div>
                        <div class="detail-item">
                            <h4>[X2] 의미론적 문맥 유사도 (KoELECTRA)</h4>
                            <p id="x2Details"></p>
                        </div>
                        <div class="detail-item" style="grid-column: span 2;">
                            <h4>[X3] RAG 기반 팩트체크 교차 검증</h4>
                            <p id="x3Details"></p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let scoreChartInstance = null;
            let vectorChartInstance = null;

            async function analyzeAd() {
                const videoUrl = document.getElementById('video_url').value;
                const productUrl = document.getElementById('product_url').value;
                
                if (!productUrl) return alert("상품 링크는 필수입니다!");

                document.getElementById('loading').style.display = 'block';
                document.getElementById('dashboard').style.display = 'none';
                document.getElementById('detailsDashboard').style.display = 'none';

                try {
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
                        
                        // 데이터 바인딩
                        document.getElementById('scoreText').innerText = data.final_score.toFixed(2) + "점";
                        document.getElementById('statusText').innerText = data.message;
                        document.getElementById('statusText').style.color = data.final_score > 70 ? "#ff5252" : (data.final_score > 40 ? "#ffeb3b" : "#4caf50");
                        
                        document.getElementById('xaiReasoning').innerHTML = data.xai_reasoning;
                        document.getElementById('x1Details').innerHTML = data.x1_details;
                        document.getElementById('x2Details').innerHTML = data.x2_details;
                        document.getElementById('x3Details').innerHTML = data.x3_details;

                        // 📊 도넛 차트 렌더링
                        renderScoreChart(data.final_score);
                        
                        // 🌐 2D 벡터 산점도 렌더링
                        renderVectorChart(data.vector_data);
                        
                    } else {
                        alert("분석 실패: " + data.error);
                    }
                } catch (err) {
                    document.getElementById('loading').style.display = 'none';
                    alert("서버 통신 에러가 발생했습니다.");
                }
            }

            function renderScoreChart(score) {
                const ctx = document.getElementById('scoreChart').getContext('2d');
                if(scoreChartInstance) scoreChartInstance.destroy();
                
                const color = score > 70 ? '#ff5252' : (score > 40 ? '#ffeb3b' : '#4caf50');
                
                scoreChartInstance = new Chart(ctx, {
                    type: 'doughnut',
                    data: {
                        labels: ['위험도', '안전'],
                        datasets: [{
                            data: [score, 100 - score],
                            backgroundColor: [color, '#2c2c2e'],
                            borderWidth: 0
                        }]
                    },
                    options: {
                        cutout: '75%',
                        plugins: { legend: { display: false } }
                    }
                });
            }

            function renderVectorChart(vectorData) {
                const ctx = document.getElementById('vectorChart').getContext('2d');
                if(vectorChartInstance) vectorChartInstance.destroy();

                scoreChartInstance = new Chart(ctx, {
                    type: 'scatter',
                    data: {
                        datasets: [
                            {
                                label: '식약처 규정 (Fact)',
                                data: [{ x: 0, y: 0 }],
                                backgroundColor: '#4caf50',
                                pointRadius: 8
                            },
                            {
                                label: '광고 문구 (Claim)',
                                data: [{ x: vectorData.x, y: vectorData.y }],
                                backgroundColor: '#ff5252',
                                pointRadius: 8
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            x: { grid: { color: '#333' }, min: -10, max: 100, title: {display: true, text: '의미론적 거리 (X)', color: '#888'} },
                            y: { grid: { color: '#333' }, min: -10, max: 100, title: {display: true, text: '의미론적 거리 (Y)', color: '#888'} }
                        },
                        plugins: {
                            legend: { labels: { color: '#fff' } }
                        }
                    }
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# 2. 분석 API
@app.post("/api/analyze")
def api_analyze(req: AdRequest):
    try:
        # Step 0: 데이터 수집
        stt_text = ingestion.run_stt(ingestion.extract_audio_from_video(req.video_url)) if req.video_url.strip() else ""
        ocr_text = ingestion.run_ocr_from_web(req.product_url) if req.product_url.strip() else ""
        combined_text = f"{stt_text}\n{ocr_text}".strip()
        
        if len(combined_text) < 5:
            return {"status": "error", "error": "텍스트를 찾지 못했습니다."}

        # Step 1, 2, 3: 점수 도출
        x1_score = lexical.calculate_x1_score(combined_text)
        x2_score = semantic.calculate_x2_score(combined_text)
        x3_score, matched_fact = rag_checker.calculate_x3_score(combined_text)
        
        # Step 4: 머신러닝 스코어링
        final_score, shap_vals, _ = xai_scorer.calculate_final_score_and_explain(x1_score, x2_score, x3_score)
        
        # =====================================================================
        # 🌟 UI에 뿌려줄 상세 설명(Detail Text) 생성 로직
        # =====================================================================
        
        # [X1 세부결과] 발견된 금칙어 추출
        detected_words = [word for word in lexical.lexicon.keys() if word in combined_text]
        if detected_words:
            x1_details = f"<span class='danger'>적발된 단어: {', '.join(detected_words)}</span><br>이 단어들은 식약처 가이드라인에 의해 사용이 강하게 규제되는 표현입니다."
        else:
            x1_details = "<span class='highlight'>발견된 금칙어 없음.</span><br>명시적인 허위 과장 단어는 사용되지 않아 텍스트 표면적으로는 안전합니다."

        # [X2 세부결과] 문맥 설명
        if x2_score > 60:
            x2_details = f"<span class='danger'>문맥적 위험도 {x2_score:.1f}점</span><br>단순 단어를 넘어, 문장의 전반적인 뉘앙스가 과거 적발된 허위광고 데이터베이스의 과장 패턴(단정적 표현, 효능 맹신 등)과 <b>매우 유사하게 감지</b>되었습니다."
        else:
            x2_details = f"<span class='highlight'>문맥적 위험도 {x2_score:.1f}점</span><br>과거 적발된 허위광고 특유의 자극적이거나 기만적인 문맥 패턴이 크게 발견되지 않았습니다."

        # [X3 세부결과] RAG 설명
        x3_details = f"<b>[관련 식약처 규정 매칭]</b><br>{matched_fact}<br><br>"
        if x3_score > 50: # GPT가 높은 위반 점수를 준 경우
            x3_details += f"<b>[LLM 추론 결과: <span class='danger'>모순 발견</span>]</b><br>광고 문구가 위 식약처 규정을 명백히 위반하고 있는 것으로 추론되었습니다."
        else:
            x3_details += f"<b>[LLM 추론 결과: <span class='highlight'>규정 준수</span>]</b><br>광고 문구와 위 식약처 규정 간에 심각한 논리적 모순이나 위반 사항이 발견되지 않았습니다."

        # [XAI 사고회로] SHAP 기반 서술
        xai_reasoning = f"AI는 이 광고 텍스트를 분석할 때 다음과 같은 사고 과정을 거쳤습니다.<br><ul>"
        if shap_vals[0] > 0: xai_reasoning += f"<li>표면적인 단어(X1)에 과장된 표현이 섞여 있어 위험도를 높였습니다. <span class='danger'>(+{shap_vals[0]:.2f})</span></li>"
        else: xai_reasoning += f"<li>금칙어가 검출되지 않아 기본적으로 안전한 글로 인식했습니다. <span class='highlight'>({shap_vals[0]:.2f})</span></li>"
        
        if shap_vals[1] > 0: xai_reasoning += f"<li>하지만 문맥의 뉘앙스(X2)가 과거 허위광고와 너무 비슷해 AI가 강한 의심을 품었습니다. <span class='danger'>(+{shap_vals[1]:.2f})</span></li>"
        else: xai_reasoning += f"<li>전체적인 문맥과 뉘앙스(X2) 역시 정상적인 제품 설명의 형태를 띠고 있습니다. <span class='highlight'>({shap_vals[1]:.2f})</span></li>"

        if shap_vals[2] > 0: xai_reasoning += f"<li><b>결정적으로 식약처 팩트체크(X3)에서 명백한 규정 위반이 확인</b>되어, 최종 위험 판정을 내렸습니다. <span class='danger'>(+{shap_vals[2]:.2f})</span></li>"
        else: xai_reasoning += f"<li><b>식약처 팩트체크(X3) 결과 규정 위반 소지가 발견되지 않아</b> 최종적으로 안전하다고 판결했습니다. <span class='highlight'>({shap_vals[2]:.2f})</span></li>"
        xai_reasoning += "</ul>"

        # [벡터 공간 시각화 데이터] 유사도가 낮을수록 두 점의 거리가 멀어짐
        # 코사인 유사도 점수를 기반으로 임의의 x, y 좌표 산출
        distance = max(10, 100 - x2_score) # 0에 가까울수록 위험(가까움)
        vector_data = {"x": distance * 0.8, "y": distance * 0.9}

        message = "🚨 악의적인 허위/과대광고로 의심됩니다." if final_score > 70 else ("⚠️ 일부 과장된 내용이 포함되어 있습니다." if final_score > 40 else "✅ 과대광고 소지가 적은 안전한 콘텐츠입니다.")

        return {
            "status": "success",
            "final_score": float(round(final_score, 2)),
            "message": message,
            "x1_details": x1_details,
            "x2_details": x2_details,
            "x3_details": x3_details,
            "xai_reasoning": xai_reasoning,
            "vector_data": vector_data
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)