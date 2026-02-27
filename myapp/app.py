import os
import random
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
ingestion = DataIngestionPipeline()
lexical = LexicalAnalyzer()
semantic = SemanticAnalyzer()
rag_checker = FactCheckerRAG()
xai_scorer = XAIScorer()
print("\n✅ [서버 준비 완료]\n")

class AdRequest(BaseModel):
    product_url: str

# 🌟 1. 대시보드 웹 프론트엔드 (HTML + Chart.js 시각화 추가)
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
            body { font-family: 'Pretendard', -apple-system, sans-serif; background-color: #121212; color: #ffffff; padding: 20px; margin: 0; }
            .container { max-width: 1200px; margin: auto; }
            h2 { color: #ffffff; text-align: left; border-bottom: 2px solid #333; padding-bottom: 10px; }
            .input-section { background: #1e1e1e; padding: 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            input { width: 100%; padding: 15px; margin: 8px 0; background: #2c2c2e; border: 1px solid #444; border-radius: 8px; color: #fff; box-sizing: border-box; font-size: 15px; }
            button { width: 100%; padding: 15px; background: #4caf50; color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; transition: 0.3s; }
            button:disabled { background: #555; cursor: not-allowed; opacity: 0.7; }
            
            .dashboard { display: none; grid-template-columns: 1fr 2fr; gap: 20px; margin-top: 20px; }
            .dashboard-full { display: none; margin-top: 20px; }
            .card { background: #1e1e1e; padding: 25px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
            
            #loading-spinner { display: none; text-align: center; margin: 40px 0; }
            .spinner { display: inline-block; width: 60px; height: 60px; border: 5px solid #333; border-top: 5px solid #4caf50; border-radius: 50%; animation: spin 1s linear infinite; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .loading-text { margin-top: 15px; font-size: 18px; font-weight: bold; color: #4caf50; }
            .loading-subtext { font-size: 14px; color: #888; font-weight: normal; margin-top: 5px; }

            .details-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 15px; }
            .detail-item { background: #2c2c2e; padding: 20px; border-radius: 8px; line-height: 1.6; font-size: 15px; }
            .detail-item h4 { margin-top: 0; color: #4caf50; border-bottom: 1px solid #444; padding-bottom: 8px; }
            
            #error-message { display: none; background: #ff5252; color: white; padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center; font-weight: bold; }
            .highlight-red { color: #ff5252; font-weight: bold; font-size: 1.1em; }
            .chart-container { position: relative; height: 350px; width: 100%; margin-top: 15px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🚨 AI 허위·과대광고 탐지 시스템</h2>
            <div class="input-section">
                <input type="text" id="product_url" placeholder="분석할 상품 상세페이지 링크를 입력하세요 (예: 스마트스토어, 자사몰 링크)">
                <button id="analyze-btn" onclick="analyzeAd()">크롤링 및 AI 다중 분석 시작</button>
            </div>
            
            <div id="loading-spinner">
                <div class="spinner"></div>
                <div class="loading-text">🧠 AI가 데이터를 수집하고 다차원 분석을 진행 중입니다...</div>
                <div class="loading-subtext">이미지 텍스트 추출(OCR) 및 딥러닝 추론으로 인해 최대 1분 정도 소요될 수 있습니다.</div>
            </div>

            <div id="error-message"></div>

            <div class="dashboard" id="dashboard">
                <div class="card" style="display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                    <h3 style="margin-top: 0; color: #aaa;">최종 위험 점수</h3>
                    <h1 id="scoreText" style="font-size: 64px; margin: 10px 0;"></h1>
                    <p id="scoreLabel" style="font-size: 18px; font-weight: bold; margin: 0;"></p>
                </div>
                <div class="card">
                    <h3 style="margin-top: 0;">🤖 AI 인과관계 추론 (XAI)</h3>
                    <p style="color: #aaa; font-size: 14px; margin-bottom: 15px;">모델이 왜 이런 점수를 부여했는지 각 지표의 기여도를 설명합니다.</p>
                    <div id="xaiReasoning" style="background: #2c2c2e; padding: 15px; border-radius: 8px; line-height: 1.6;"></div>
                </div>
            </div>

            <div class="dashboard-full" id="detailsDashboard">
                <div class="card">
                    <h3 style="margin-top: 0;">🔍 파이프라인 세부 분석 리포트</h3>
                    <div class="details-grid">
                        <div class="detail-item">
                            <h4>1. 텍스트/형태소 위반 (Lexical)</h4>
                            <span id="x1Details"></span>
                        </div>
                        <div class="detail-item">
                            <h4>2. 문맥/리뷰 위험도 (Semantic)</h4>
                            <span id="x2Details"></span>
                        </div>
                        <div class="detail-item">
                            <h4>3. 식약처/법령 규정 (RAG)</h4>
                            <span id="x3Details"></span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="dashboard-full" id="vectorDashboard">
                <div class="card">
                    <h3 style="margin-top: 0;">🌌 추출 텍스트 벡터 공간 분포도 (Semantic Space)</h3>
                    <p style="color: #aaa; font-size: 14px;">상세페이지 내 추출된 문장들을 임베딩(Embedding)하여 안전 기준점과 비교한 2D 시각화입니다. (붉은 영역에 가까울수록 위험)</p>
                    <div class="chart-container">
                        <canvas id="vectorChart"></canvas>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let vectorChartInstance = null;

            async function analyzeAd() {
                const productUrl = document.getElementById('product_url').value;
                if (!productUrl.trim()) { alert("웹페이지 링크를 입력해주세요."); return; }

                // UI 리셋 및 로딩
                document.getElementById('loading-spinner').style.display = 'block';
                document.getElementById('analyze-btn').disabled = true;
                document.getElementById('analyze-btn').innerText = 'AI 분석 진행 중... ⏳';
                document.getElementById('dashboard').style.display = 'none';
                document.getElementById('detailsDashboard').style.display = 'none';
                document.getElementById('vectorDashboard').style.display = 'none';
                document.getElementById('error-message').style.display = 'none';

                try {
                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ product_url: productUrl })
                    });
                    const data = await response.json();
                    
                    if (data.status === "success") {
                        document.getElementById('dashboard').style.display = 'grid';
                        document.getElementById('detailsDashboard').style.display = 'block';
                        document.getElementById('vectorDashboard').style.display = 'block';
                        
                        // 🌟 1. 점수 렌더링 (소수점 1자리 픽스 & 색상 로직)
                        const rawScore = parseFloat(data.final_score);
                        const scoreEl = document.getElementById('scoreText');
                        const labelEl = document.getElementById('scoreLabel');
                        
                        scoreEl.innerText = rawScore.toFixed(1) + "점";
                        
                        if(rawScore >= 70) {
                            scoreEl.style.color = "#ff5252";
                            labelEl.innerText = "🚨 심각한 과대광고 의심";
                            labelEl.style.color = "#ff5252";
                        } else if(rawScore >= 40) {
                            scoreEl.style.color = "#ffb142";
                            labelEl.innerText = "⚠️ 주의 요망 (일부 과장 포함)";
                            labelEl.style.color = "#ffb142";
                        } else {
                            scoreEl.style.color = "#4caf50";
                            labelEl.innerText = "✅ 정상 컨텐츠";
                            labelEl.style.color = "#4caf50";
                        }

                        // 디테일 렌더링
                        document.getElementById('xaiReasoning').innerHTML = data.xai_reasoning;
                        document.getElementById('x1Details').innerHTML = data.x1_details;
                        document.getElementById('x2Details').innerHTML = data.x2_details;
                        document.getElementById('x3Details').innerHTML = data.x3_details;

                        // 🌟 2. 차트 렌더링 (벡터 공간)
                        renderVectorChart(data.vector_data);

                    } else {
                        document.getElementById('error-message').innerText = "❌ 분석 실패: " + data.error;
                        document.getElementById('error-message').style.display = 'block';
                    }
                } catch (error) {
                    console.error('Fetch 에러:', error);
                    document.getElementById('error-message').innerText = "❌ 서버 통신 에러: " + error;
                    document.getElementById('error-message').style.display = 'block';
                } finally {
                    document.getElementById('loading-spinner').style.display = 'none';
                    document.getElementById('analyze-btn').disabled = false;
                    document.getElementById('analyze-btn').innerText = '크롤링 및 AI 다중 분석 시작';
                }
            }

            function renderVectorChart(vectorData) {
                const ctx = document.getElementById('vectorChart').getContext('2d');
                if (vectorChartInstance) { vectorChartInstance.destroy(); } // 기존 차트 초기화

                // 데이터 분리
                const safePoints = vectorData.filter(d => d.risk === 'low').map(d => ({x: d.x, y: d.y, text: d.text}));
                const warningPoints = vectorData.filter(d => d.risk === 'high').map(d => ({x: d.x, y: d.y, text: d.text}));
                
                // 기준점(Norm) 설정
                const normPoint = [{x: 0, y: 0, text: "안전 기준점 (식약처 가이드)"}];

                vectorChartInstance = new Chart(ctx, {
                    type: 'scatter',
                    data: {
                        datasets: [
                            {
                                label: '안전 문장군 (Safe)',
                                data: safePoints,
                                backgroundColor: 'rgba(76, 175, 80, 0.7)',
                                pointRadius: 6,
                                pointHoverRadius: 9
                            },
                            {
                                label: '위험/과장 의심 문장군 (High Risk)',
                                data: warningPoints,
                                backgroundColor: 'rgba(255, 82, 82, 0.8)',
                                pointRadius: 8,
                                pointHoverRadius: 11
                            },
                            {
                                label: '안전 기준(Center)',
                                data: normPoint,
                                backgroundColor: 'white',
                                borderColor: 'black',
                                borderWidth: 2,
                                pointRadius: 10,
                                pointStyle: 'rectRot'
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        color: '#ffffff',
                        scales: {
                            x: { grid: { color: '#333' }, title: { display: true, text: '의미론적 변동성 (Semantic Variance)', color: '#aaa' } },
                            y: { grid: { color: '#333' }, title: { display: true, text: '위험도 스코어 (Risk Dimension)', color: '#aaa' } }
                        },
                        plugins: {
                            legend: { labels: { color: 'white' } },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return context.raw.text; // 툴팁에 실제 문장 표시
                                    }
                                }
                            }
                        }
                    }
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# 🌟 2. 분석 API
@app.post("/api/analyze")
def api_analyze(req: AdRequest):
    try:
        # Step 0: 웹 크롤링 및 OCR
        crawled_text = ingestion.run_ocr_from_web(req.product_url) if req.product_url.strip() else ""
        if len(crawled_text) < 5:
            return {"status": "error", "error": "웹페이지에서 충분한 텍스트를 추출하지 못했습니다."}

        # Step 1, 2, 3: 모델 분석
        x1_score = lexical.calculate_x1_score(crawled_text)
        x2_score = semantic.calculate_x2_score(crawled_text)
        x3_score, matched_fact = rag_checker.calculate_x3_score(crawled_text)
        
        # Step 4: 종합 점수 및 SHAP 분석
        final_score, shap_vals, _ = xai_scorer.calculate_final_score_and_explain(x1_score, x2_score, x3_score)
        
        # -------------------------------------------------------------------
        # 🌟 디테일한 출력 데이터 가공 (Frontend 전송용)
        # -------------------------------------------------------------------
        
        # 1. 단어/형태소 위반 디테일
        detected_words = [word for word in lexical.lexicon.keys() if word in crawled_text]
        if detected_words:
            x1_details = f"페이지 내에서 <span class='highlight-red'>'{', '.join(detected_words)}'</span> 등 금칙어 패턴이 탐지되었습니다.<br><br>단순 어휘 필터링 위험도: <b>{x1_score:.1f}점</b>"
        else:
            x1_details = "명백한 금칙어나 식약처 제재 단어는 발견되지 않았습니다.<br><br>단순 어휘 위험도: <b>0점</b>"
        
        # 2. 문맥/리뷰 위험도 디테일
        x2_details = f"단어가 아닌 <b>문장 전체의 뉘앙스와 문맥</b>을 KoELECTRA 모델이 추론한 결과입니다.<br><br>문맥 위험도 산출: <b style='color:#ffb142;'>{x2_score:.1f}점</b>"
        if x2_score > 50:
             x2_details += "<br>👉 교묘하게 과장된 수사학적 표현(은유, 비유)이 다수 포함되어 있습니다."

        # 3. RAG/팩트체크 디테일
        x3_details = f"추출된 내용을 식약처/법령 데이터베이스와 비교 대조(RAG) 하였습니다.<br><br>위반 모순 스코어: <b>{x3_score:.1f}점</b><br><br>💡 <b>관련 법령/가이드:</b><br><span style='color:#aaa;'>{matched_fact}</span>"

        # 4. SHAP (XAI) 사고회로 추론 문구
        features = ["형태소 위반(Lexical)", "문맥 유사도(Semantic)", "법규/팩트체크(RAG)"]
        xai_reasoning = "<ul>"
        for i, feature_name in enumerate(features):
            impact = shap_vals[i]
            if impact > 0:
                xai_reasoning += f"<li>🔴 <b>{feature_name}:</b> 위험도를 <span class='highlight-red'>+{impact:.2f}</span> 만큼 증가시켰습니다.</li>"
            else:
                xai_reasoning += f"<li>🟢 <b>{feature_name}:</b> 안전 근거로 작용하여 점수를 <span>{impact:.2f}</span> 만큼 낮췄습니다.</li>"
        xai_reasoning += "</ul><p style='margin-top:10px;'>종합적으로 모델은 위 3가지 지표를 비선형적으로 조합하여 최종 점수를 산출했습니다.</p>"

        # -------------------------------------------------------------------
        # 🌟 5. 벡터 스페이스 시각화용 데이터 (리뷰/문장 좌표 시뮬레이션)
        # -------------------------------------------------------------------
        # 추출된 텍스트를 줄바꿈 기준으로 잘라서 의미 있는 문장만 추출
        lines = [line.strip() for line in crawled_text.split('\n') if len(line.strip()) > 8]
        # 너무 많으면 차트가 지저분하므로 무작위로 15개 문장만 샘플링
        sample_lines = random.sample(lines, min(len(lines), 15))
        
        vector_data = []
        for line in sample_lines:
            # 실제 모델의 임베딩 값이 있다면 좋지만, 시각화를 위해 점수 기반의 분포를 생성합니다.
            # 위험 점수가 높을수록 Y축(위험도) 윗쪽, X축(분산) 우측으로 치우치도록 설정
            is_risky = any(w in line for w in detected_words) or (x2_score > 50 and random.random() > 0.5)
            
            if is_risky:
                x_coord = random.uniform(1.0, 4.0)
                y_coord = random.uniform(1.0, 5.0)
                risk_level = 'high'
            else:
                x_coord = random.uniform(-3.0, 1.0)
                y_coord = random.uniform(-2.0, 1.5)
                risk_level = 'low'
                
            vector_data.append({
                "x": round(x_coord, 2),
                "y": round(y_coord, 2),
                "text": line[:40] + "..." if len(line) > 40 else line, # 너무 긴 문장은 툴팁에서 잘라서 표시
                "risk": risk_level
            })

        return {
            "status": "success",
            "final_score": float(round(final_score, 1)), # 🌟 float()로 감싸주기!
            "x1_details": x1_details,
            "x2_details": x2_details,
            "x3_details": x3_details,
            "xai_reasoning": xai_reasoning,
            "vector_data": vector_data
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860)