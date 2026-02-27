from step0_ingestion import DataIngestionPipeline
from step1_lexical import LexicalAnalyzer
from step2_semantic import SemanticAnalyzer
from step3_rag import FactCheckerRAG
from step4_xai import XAIScorer # 🌟 Step 4 추가!

def run_full_pipeline():
    print("==================================================")
    print(" 🚀 [AtoZ 파이프라인] 과대광고 탐지 AI 시스템 가동 (크롤링/OCR 전용)")
    print("==================================================\n")

    # 1. 모델 및 엔진 초기화
    ingestion = DataIngestionPipeline()
    lexical = LexicalAnalyzer()
    semantic = SemanticAnalyzer()
    rag_checker = FactCheckerRAG()
    xai_scorer = XAIScorer() # 🌟 XAI 엔진 가동!

    # 🚨 유튜브 URL 제거, 상품 상세페이지 링크만 유지
    target_product_url = "https://brand.naver.com/pacsafe/products/9365045491"

    # [Step 0] 데이터 수집 (웹 크롤링/OCR 단일 파이프라인)
    print("\n▶️ [Step 0] 웹페이지 텍스트 및 이미지 데이터 수집 중...")
    
    crawled_text = ""
    try:
        # 변경된 step0 구조에 맞춰 run_ocr_from_web만 실행
        crawled_text = ingestion.run_ocr_from_web(target_product_url)
    except Exception as e:
        print(f"⚠️ OCR 추출 실패: {e}")

    # 공백이나 None 처리 로직 추가
    if not crawled_text or len(crawled_text.strip()) == 0:
        print("❌ 분석할 텍스트가 없어 종료합니다.")
        return

    # [Step 1, 2, 3] 텍스트 심층 분석
    print("\n▶️ [Step 1, 2, 3] 텍스트 심층 분석 가동...")
    
    # stt_text가 빠졌으므로 combined_text 대신 crawled_text를 그대로 사용
    x1_score = lexical.calculate_x1_score(crawled_text)
    x2_score = semantic.calculate_x2_score(crawled_text)
    x3_score, matched_fact = rag_checker.calculate_x3_score(crawled_text)
    
    # 🌟 [Step 4] XGBoost 스코어링 및 SHAP 분석
    print("\n▶️ [Step 4] XGBoost 기반 최종 스코어링 및 SHAP 설명 생성...")
    final_score, shap_vals, base_value = xai_scorer.calculate_final_score_and_explain(x1_score, x2_score, x3_score)

    # =====================================================================
    # 🌟 최종 분석 리포트 (XAI 설명 포함)
    # =====================================================================
    print("\n==================================================")
    print(" 📜 [최종 과대광고 탐지 리포트]")
    print("==================================================")
    print(f" 🔹 [X1] 형태소/단어 위반 (Lexical) : {x1_score:5.1f} 점")
    print(f" 🔹 [X2] 문맥적 유사도 (Semantic)   : {x2_score:5.1f} 점")
    print(f" 🔹 [X3] 식약처 팩트체크 (RAG)      : {x3_score:5.1f} 점")
    print("--------------------------------------------------")
    print(f" 🎯 [S] 머신러닝 최종 위험도 점수   : {final_score:5.1f} / 100.0 점")
    print("==================================================")
    
    if final_score > 70:
        print(" 🚨 [판정] 매우 위험! 악의적인 허위/과대광고로 의심됩니다.")
    elif final_score > 40:
        print(" ⚠️ [판정] 주의! 일부 과장된 표현이나 사실과 다른 내용이 포함되어 있습니다.")
    else:
        print(" ✅ [판정] 안전! 과대광고 소지가 적은 정상적인 콘텐츠입니다.")
        
    print("\n==================================================")
    print(" 🤖 [XAI] 인공지능의 판정 사유 (SHAP Values)")
    print("==================================================")
    print(" (점수가 양수(+)면 위험도를 높였고, 음수(-)면 안전하다고 판단한 근거입니다.)\n")
    
    features = ["X1 (단어 위반)", "X2 (문맥 유사도)", "X3 (팩트체크 모순)"]
    for i, feature_name in enumerate(features):
        impact = shap_vals[i]
        direction = "🔴 위험도 증가" if impact > 0 else "🟢 위험도 감소"
        # 직관성을 위해 SHAP Log-odds 값을 점수 스케일처럼 비례해서 보여줍니다
        print(f"  {direction} 기여: {feature_name:<18} -> 기여도: {impact:+.2f}")
    print("==================================================")

if __name__ == "__main__":
    run_full_pipeline()