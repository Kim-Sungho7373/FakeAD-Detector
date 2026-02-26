# 1. 🌟 대시보드 웹 프론트엔드 (HTML)
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
            
            /* 🌟 로딩 스피너 CSS 추가 */
            #loading-spinner { display: none; text-align: center; margin: 30px 0; }
            .spinner { display: inline-block; width: 50px; height: 50px; border: 5px solid #333; border-top: 5px solid #4caf50; border-radius: 50%; animation: spin 1s linear infinite; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .loading-text { margin-top: 15px; font-size: 16px; font-weight: bold; color: #4caf50; }
            .loading-subtext { font-size: 14px; color: #888; font-weight: normal; }

            .danger { color: #ff5252; font-weight: bold; }
            .highlight { color: #4caf50; font-weight: bold; }
            .details-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
            .detail-item { background: #2c2c2e; padding: 15px; border-radius: 8px; }
            #error-message { display: none; background: #ff5252; color: white; padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🚨 AI 과대광고 탐지 대시보드 (Overview)</h2>
            <div class="input-section">
                <input type="text" id="video_url" placeholder="유튜브 영상 링크 (현재 클라우드 환경에서는 비워두고 테스트 권장)">
                <input type="text" id="product_url" placeholder="상품 상세페이지 링크 (필수)">
                <button id="analyze-btn" onclick="analyzeAd()">분석 시작</button>
            </div>
            
            <div id="loading-spinner">
                <div class="spinner"></div>
                <p class="loading-text">
                    🧠 AI가 열심히 팩트체크를 진행 중입니다...<br>
                    <span class="loading-subtext">(딥러닝 분석으로 인해 약 1~2분 정도 소요될 수 있습니다)</span>
                </p>
            </div>

            <div id="error-message"></div>

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
                
                // UI 요소 가져오기
                const loadingSpinner = document.getElementById('loading-spinner');
                const analyzeBtn = document.getElementById('analyze-btn');
                const dashboard = document.getElementById('dashboard');
                const detailsDashboard = document.getElementById('detailsDashboard');
                const errorMsg = document.getElementById('error-message');

                // 🌟 로딩 UI 켜기 & 버튼 잠금
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
                        body: JSON.stringify({ video_url: videoUrl, product_url: productUrl })
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
                        // 백엔드에서 에러를 반환한 경우
                        errorMsg.innerText = "❌ 분석 실패: " + data.error;
                        errorMsg.style.display = 'block';
                    }
                } catch (error) {
                    console.error('Fetch 에러:', error);
                    errorMsg.innerText = "❌ 서버와 통신하는 중 문제가 발생했습니다.";
                    errorMsg.style.display = 'block';
                } finally {
                    // 🌟 로딩 끄기 & 버튼 원상복구
                    loadingSpinner.style.display = 'none';
                    analyzeBtn.disabled = false;
                    analyzeBtn.innerText = '분석 시작';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

