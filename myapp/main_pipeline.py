from step0_ingestion import DataIngestionPipeline
from step1_lexical import LexicalAnalyzer
from step2_semantic import SemanticAnalyzer
from step3_rag import FactCheckerRAG
from step4_xai import XAIScorer

def run_full_pipeline():
    print("==================================================")
    print(" 🚀 [End-to-End Pipeline] Launching the misleading-ad detection AI system (crawling/OCR only)")
    print("==================================================\n")

    # 1. 모델 및 엔진 초기화
    ingestion = DataIngestionPipeline()
    lexical = LexicalAnalyzer()
    semantic = SemanticAnalyzer()
    rag_checker = FactCheckerRAG()
    xai_scorer = XAIScorer()

    target_product_url = "https://brand.naver.com/pacsafe/products/9365045491"

    print("\n▶️ [Step 0] Collecting webpage text and image data...")
    
    crawled_text = ""
    try:
        crawled_text = ingestion.run_ocr_from_web(target_product_url)
    except Exception as e:
        print(f"⚠️ OCR extraction failed: {e}")

    if not crawled_text or len(crawled_text.strip()) == 0:
        print("❌ No analyzable text was found. Exiting.")
        return

    print("\n▶️ [Step 1, 2, 3] Running deep text analysis...")
    
    x1_score = lexical.calculate_x1_score(crawled_text)
    x2_score = semantic.calculate_x2_score(crawled_text)
    x3_score, matched_fact = rag_checker.calculate_x3_score(crawled_text)
    
    print("\n▶️ [Step 4] Generating final XGBoost scoring and SHAP explanations...")
    final_score, shap_vals, base_value = xai_scorer.calculate_final_score_and_explain(x1_score, x2_score, x3_score)

    print("\n==================================================")
    print(" 📜 [Final Misleading-Ad Detection Report]")
    print("==================================================")
    print(f" 🔹 [X1] Lexical trigger score       : {x1_score:5.1f} pts")
    print(f" 🔹 [X2] Semantic context score      : {x2_score:5.1f} pts")
    print(f" 🔹 [X3] Regulatory fact-check score : {x3_score:5.1f} pts")
    print("--------------------------------------------------")
    print(f" 🎯 [S] Final ML risk score          : {final_score:5.1f} / 100.0 pts")
    print("==================================================")
    
    if final_score > 70:
        print(" 🚨 [Verdict] Very high risk. Strong signs of misleading or exaggerated advertising were detected.")
    elif final_score > 40:
        print(" ⚠️ [Verdict] Caution. Some exaggerated or potentially inaccurate claims were detected.")
    else:
        print(" ✅ [Verdict] Low risk. The content appears relatively safe.")
        
    print("\n==================================================")
    print(" 🤖 [XAI] Why the model made this decision (SHAP values)")
    print("==================================================")
    print(" (Positive values increase risk, while negative values support a safer assessment.)\n")
    
    features = ["X1 (Lexical)", "X2 (Semantic)", "X3 (Fact Check)"]
    for i, feature_name in enumerate(features):
        impact = shap_vals[i]
        direction = "🔴 Risk increase" if impact > 0 else "🟢 Risk decrease"
        print(f"  {direction} contribution: {feature_name:<18} -> impact: {impact:+.2f}")
    print("==================================================")

if __name__ == "__main__":
    run_full_pipeline()
