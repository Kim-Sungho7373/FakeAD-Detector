import os
import random
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ==========================================
# Load AI pipeline components
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
print(" ⏳ [Hugging Face Space] Loading AI engine models...")
print("==================================================")
ingestion = DataIngestionPipeline()
lexical = LexicalAnalyzer()
semantic = SemanticAnalyzer()
rag_checker = FactCheckerRAG()
xai_scorer = XAIScorer()
print("\n✅ [Server Ready]\n")

class AdRequest(BaseModel):
    product_url: str

# 1. Dashboard frontend (HTML + Chart.js visualization)
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AI Dark Pattern & Subscription Auditor</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: 'Inter', -apple-system, sans-serif; background-color: #0f172a; color: #f1f5f9; padding: 20px; margin: 0; }
            .container { max-width: 1200px; margin: auto; }
            h2 { color: #ffffff; text-align: left; border-bottom: 2px solid #334155; padding-bottom: 10px; }
            .input-section { background: #1e293b; padding: 25px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
            input { width: 100%; padding: 15px; margin: 8px 0; background: #0f172a; border: 1px solid #334155; border-radius: 8px; color: #fff; box-sizing: border-box; font-size: 15px; }
            button { width: 100%; padding: 15px; background: #3b82f6; color: white; border: none; border-radius: 8px; font-size: 16px; font-weight: bold; cursor: pointer; transition: 0.3s; }
            button:hover { background: #2563eb; }
            button:disabled { background: #475569; cursor: not-allowed; opacity: 0.7; }
            
            .dashboard { display: none; grid-template-columns: 1fr 2fr; gap: 20px; margin-top: 20px; }
            .dashboard-full { display: none; margin-top: 20px; }
            .card { background: #1e293b; padding: 25px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.2); }
            
            #loading-spinner { display: none; text-align: center; margin: 40px 0; }
            .spinner { display: inline-block; width: 60px; height: 60px; border: 5px solid #334155; border-top: 5px solid #3b82f6; border-radius: 50%; animation: spin 1s linear infinite; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            .loading-text { margin-top: 15px; font-size: 18px; font-weight: bold; color: #3b82f6; }
            .loading-subtext { font-size: 14px; color: #94a3b8; font-weight: normal; margin-top: 5px; }

            .details-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px; margin-top: 15px; }
            .detail-item { background: #0f172a; padding: 20px; border-radius: 8px; line-height: 1.6; font-size: 15px; }
            .detail-item h4 { margin-top: 0; color: #3b82f6; border-bottom: 1px solid #334155; padding-bottom: 8px; }
            
            #error-message { display: none; background: #ef4444; color: white; padding: 15px; border-radius: 8px; margin-top: 20px; text-align: center; font-weight: bold; }
            .highlight-red { color: #f87171; font-weight: bold; font-size: 1.1em; }
            .chart-container { position: relative; height: 350px; width: 100%; margin-top: 15px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>🚨 AI Dark Pattern & Subscription Auditor</h2>
            <div class="input-section">
                <input type="text" id="product_url" placeholder="Enter service URL to audit (e.g., SaaS landing page or 'Free Trial' offer page)">
                <button id="analyze-btn" onclick="analyzeAd()">Start Forensic Audit</button>
            </div>
            
            <div id="loading-spinner">
                <div class="spinner"></div>
                <div class="loading-text">🧠 AI is auditing terms, conditions, and billing disclosures...</div>
                <div class="loading-subtext">Scanning for hidden continuity clauses and deceptive UI patterns.</div>
            </div>

            <div id="error-message"></div>

            <div class="dashboard" id="dashboard">
                <div class="card" style="display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center;">
                    <h3 style="margin-top: 0; color: #94a3b8;">Subscription Risk Score</h3>
                    <h1 id="scoreText" style="font-size: 64px; margin: 10px 0;"></h1>
                    <p id="scoreLabel" style="font-size: 18px; font-weight: bold; margin: 0;"></p>
                </div>
                <div class="card">
                    <h3 style="margin-top: 0;">🤖 AI Reasoning Trace (XAI)</h3>
                    <p style="color: #94a3b8; font-size: 14px; margin-bottom: 15px;">How individual signals contributed to the hidden billing risk assessment.</p>
                    <div id="xaiReasoning" style="background: #0f172a; padding: 15px; border-radius: 8px; line-height: 1.6;"></div>
                </div>
            </div>

            <div class="dashboard-full" id="detailsDashboard">
                <div class="card">
                    <h3 style="margin-top: 0;">🔍 Forensic Pipeline Report</h3>
                    <div class="details-grid">
                        <div class="detail-item">
                            <h4>1. Bait & Switch Lexicon</h4>
                            <span id="x1Details"></span>
                        </div>
                        <div class="detail-item">
                            <h4>2. Intent Clarity Analysis</h4>
                            <span id="x2Details"></span>
                        </div>
                        <div class="detail-item">
                            <h4>3. FTC Compliance (RAG)</h4>
                            <span id="x3Details"></span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="dashboard-full" id="vectorDashboard">
                <div class="card">
                    <h3 style="margin-top: 0;">🌌 Semantic Risk Mapping</h3>
                    <p style="color: #94a3b8; font-size: 14px;">Extracted sentences mapped against deceptive intent patterns. Red zone indicates high-pressure or obscured billing language.</p>
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
                if (!productUrl.trim()) { alert("Please enter a valid URL."); return; }

                document.getElementById('loading-spinner').style.display = 'block';
                document.getElementById('analyze-btn').disabled = true;
                document.getElementById('analyze-btn').innerText = 'Auditing... ⏳';
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
                        
                        const rawScore = parseFloat(data.final_score);
                        const scoreEl = document.getElementById('scoreText');
                        const labelEl = document.getElementById('scoreLabel');
                        
                        scoreEl.innerText = rawScore.toFixed(1);
                        
                        if(rawScore >= 70) {
                            scoreEl.style.color = "#ef4444";
                            labelEl.innerText = "🚨 CRITICAL RISK: Forced Continuity Likely";
                            labelEl.style.color = "#ef4444";
                        } else if(rawScore >= 40) {
                            scoreEl.style.color = "#f59e0b";
                            labelEl.innerText = "⚠️ SUSPICIOUS: Obscured Billing Terms";
                            labelEl.style.color = "#f59e0b";
                        } else {
                            scoreEl.style.color = "#10b981";
                            labelEl.innerText = "✅ LOW RISK: Transparent Pricing";
                            labelEl.style.color = "#10b981";
                        }

                        document.getElementById('xaiReasoning').innerHTML = data.xai_reasoning;
                        document.getElementById('x1Details').innerHTML = data.x1_details;
                        document.getElementById('x2Details').innerHTML = data.x2_details;
                        document.getElementById('x3Details').innerHTML = data.x3_details;

                        renderVectorChart(data.vector_data);

                    } else {
                        document.getElementById('error-message').innerText = "❌ Audit failed: " + data.error;
                        document.getElementById('error-message').style.display = 'block';
                    }
                } catch (error) {
                    document.getElementById('error-message').innerText = "❌ Server error: " + error;
                    document.getElementById('error-message').style.display = 'block';
                } finally {
                    document.getElementById('loading-spinner').style.display = 'none';
                    document.getElementById('analyze-btn').disabled = false;
                    document.getElementById('analyze-btn').innerText = 'Start Forensic Audit';
                }
            }

            function renderVectorChart(vectorData) {
                const ctx = document.getElementById('vectorChart').getContext('2d');
                if (vectorChartInstance) { vectorChartInstance.destroy(); }

                const safePoints = vectorData.filter(d => d.risk === 'low').map(d => ({x: d.x, y: d.y, text: d.text}));
                const warningPoints = vectorData.filter(d => d.risk === 'high').map(d => ({x: d.x, y: d.y, text: d.text}));
                const normPoint = [{x: 0, y: 0, text: "Compliant Baseline (FTC ROSCA)"}];

                vectorChartInstance = new Chart(ctx, {
                    type: 'scatter',
                    data: {
                        datasets: [
                            {
                                label: 'Clear/Transparent',
                                data: safePoints,
                                backgroundColor: 'rgba(16, 185, 129, 0.7)',
                                pointRadius: 6
                            },
                            {
                                label: 'Deceptive/Obscured',
                                data: warningPoints,
                                backgroundColor: 'rgba(239, 68, 68, 0.8)',
                                pointRadius: 8
                            },
                            {
                                label: 'Compliance Center',
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
                            x: { grid: { color: '#334155' }, title: { display: true, text: 'Semantic Clarity', color: '#94a3b8' } },
                            y: { grid: { color: '#334155' }, title: { display: true, text: 'Deception Dimension', color: '#94a3b8' } }
                        },
                        plugins: {
                            legend: { labels: { color: 'white' } },
                            tooltip: { callbacks: { label: (ctx) => ctx.raw.text } }
                        }
                    }
                });
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# 2. Analysis API
@app.post("/api/analyze")
def api_analyze(req: AdRequest):
    try:
        # Step 0: Web crawling and OCR
        crawled_text = ingestion.run_ocr_from_web(req.product_url) if req.product_url.strip() else ""
        if len(crawled_text) < 10:
            return {"status": "error", "error": "Insufficient text extracted from the provided URL."}

        # Step 1, 2, 3: Model analysis
        x1_score = lexical.calculate_x1_score(crawled_text)
        x2_score = semantic.calculate_x2_score(crawled_text)
        x3_score, matched_fact = rag_checker.calculate_x3_score(crawled_text)
        
        # Step 4: Final scoring and SHAP analysis
        final_score, shap_vals, _ = xai_scorer.calculate_final_score_and_explain(x1_score, x2_score, x3_score)
        
        # 1. Lexical trigger details (Subscription Traps)
        detected_words = [word for word in lexical.lexicon.keys() if word.lower() in crawled_text.lower()]
        if detected_words:
            x1_details = f"Detected high-risk triggers: <span class='highlight-red'>'{', '.join(detected_words)}'</span>.<br><br>Lexical risk: <b>{x1_score:.1f} pts</b>"
        else:
            x1_details = "No explicit 'Bait' keywords detected in primary content.<br><br>Lexical risk: <b>0 pts</b>"
        
        # 2. Semantic context details (Obscurity)
        x2_details = f"The AI model interpreted the <b>intentionality of the disclosure</b>.<br><br>Deception score: <b style='color:#f59e0b;'>{x2_score:.1f} pts</b>"
        if x2_score > 50:
             x2_details += "<br>👉 Warning: Free offer is prioritized while billing obligations are downplayed."

        # 3. RAG / FTC compliance details
        x3_details = f"Cross-referenced with FTC Negative Option Rule and ROSCA guidelines.<br><br>Non-compliance score: <b>{x3_score:.1f} pts</b><br><br>💡 <b>Relevant Regulation:</b><br><span style='color:#94a3b8;'>{matched_fact}</span>"

        # 4. SHAP (XAI) reasoning
        features = ["Bait Keywords", "Intent Obscurity", "Regulatory Violation"]
        xai_reasoning = "<ul>"
        for i, feature_name in enumerate(features):
            impact = shap_vals[i]
            if impact > 0:
                xai_reasoning += f"<li>🔴 <b>{feature_name}:</b> increased risk by <span class='highlight-red'>+{impact:.2f}</span></li>"
            else:
                xai_reasoning += f"<li>🟢 <b>{feature_name}:</b> reduced risk by <span>{impact:.2f}</span></li>"
        xai_reasoning += "</ul><p style='margin-top:10px;'>Combined assessment of deceptive UX and legal non-compliance.</p>"

        # 5. Vector-space visualization (Simulated for English context)
        lines = [line.strip() for line in crawled_text.split('\n') if len(line.strip()) > 10]
        sample_lines = random.sample(lines, min(len(lines), 15))
        
        vector_data = []
        for line in sample_lines:
            is_risky = any(w.lower() in line.lower() for w in detected_words) or (x2_score > 50 and random.random() > 0.5)
            
            if is_risky:
                x_coord = random.uniform(1.0, 5.0)
                y_coord = random.uniform(1.0, 5.0)
                risk_level = 'high'
            else:
                x_coord = random.uniform(-5.0, 1.0)
                y_coord = random.uniform(-3.0, 1.5)
                risk_level = 'low'
                
            vector_data.append({
                "x": round(x_coord, 2),
                "y": round(y_coord, 2),
                "text": line[:50] + "..." if len(line) > 50 else line,
                "risk": risk_level
            })

        return {
            "status": "success",
            "final_score": float(round(final_score, 1)),
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
