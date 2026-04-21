import os
import re
import torch
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# Load OPENAI_API_KEY from the .env file.
load_dotenv()

class FactCheckerRAG:
    def __init__(self):
        print("📚 [Step 3] Loading the RAG + LLM fact checker...")
        
        # 기기 설정 (Mac의 경우 mps, 아니면 cpu)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # 한국어 문장 임베딩 모델 로드
        self.retriever = SentenceTransformer('jhgan/ko-sroberta-multitask', device=self.device)
        
        # OpenAI 클라이언트 초기화 (API 키는 환경변수에서 안전하게 가져옴)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ Warning: OPENAI_API_KEY is not set. Please check your .env file.")
        
        self.client = OpenAI(api_key=api_key)
        
        # 팩트 데이터베이스 (식약처 규정 및 판례 기반)
        self.fact_db = [
            # 1. 절대 금지 조항
            "질병의 예방 및 치료에 효능·효과가 있거나 의약품 또는 건강기능식품으로 오인·혼동할 우려가 있는 표시·광고는 금지됩니다.",
            "체험기 등을 이용하거나 '주문쇄도', '단체추천' 등 소비자를 기만하는 광고는 처벌 대상입니다.",
            "식품에 각종 상장, 인증, 보증을 받았다는 내용을 사용하는 것은 허위·과대광고에 해당할 수 있습니다.",
            
            # 2. 허용되는 표현
            "인체의 건전한 성장 및 발달과 건강 유지에 도움을 준다는 표현은 특정 질병을 언급하지 않는 한 허용됩니다.",
            "건강증진, 체질개선, 식이요법, 영양보급 등에 도움을 준다는 표현은 과대광고가 아닙니다.",
            "해당 제품이 유아식, 환자식 등 특수용도식품이라는 표현은 허용됩니다.",
            
            # 3. 실제 적발 사례
            "일반 식품에 당뇨, 고혈압, 항암 등 특정 질병 치료 효과가 있다고 기재하는 것은 명백한 불법입니다.",
            "블로그나 쇼핑몰에 질병 치료 전후 비교 사진이나 개인적인 체험기를 올리는 행위는 불법 과대광고입니다."
        ]
        
        # 데이터베이스 임베딩 미리 계산
        self.db_embeddings = self.retriever.encode(self.fact_db, convert_to_tensor=True)

    def calculate_x3_score(self, text):
        """Calculate a regulatory violation score for ad copy via RAG."""
        if not text or len(text.strip()) < 5:
            return 0.0, "Not enough text was provided for inspection."

        try:
            # 1. 관련 규정 검색 (Retrieval)
            query_embedding = self.retriever.encode(text, convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_embedding, self.db_embeddings)[0]
            best_idx = torch.argmax(cosine_scores).item()
            retrieved_fact = self.fact_db[best_idx]
            
            # 2. LLM 심사 (Generation)
            prompt = f"""
            You are a regulatory reviewer for misleading food and advertising claims in Korea.
            Based on the [Relevant Regulation] below, assess whether the [Ad Copy] is non-compliant.

            [Relevant Regulation]: {retrieved_fact}
            [Ad Copy]: {text}

            Reply only in the following format:
            Score: [number between 0 and 100]
            Reason: [1-2 lines explaining the specific basis for non-compliance, or why the claim appears acceptable]
            """
            
            print("   -> 🤖 GPT reviewer is analyzing the content...")
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content
            print(f"   [Result] {result_text}")
            
            score_match = re.search(r"Score:\s*(\d+)", result_text)
            x3_score = float(score_match.group(1)) if score_match else 0.0
            
            return x3_score, retrieved_fact
            
        except Exception as e:
            print(f"⚠️ Error occurred: {e}")
            return 0.0, "An error occurred during analysis."

# 앙상블 점수 계산기
def calculate_final_score(x1, x2, x3):
    """
    x1: 키워드 매칭 점수
    x2: 딥러닝 문맥 점수
    x3: RAG 팩트체크 점수
    """
    w1, w2, w3 = 0.2, 0.4, 0.4
    return (w1 * x1) + (w2 * x2) + (w3 * x3)

if __name__ == "__main__":
    checker = FactCheckerRAG()
    test_ad = "Drink this tea and not only prevent cancer, but also instantly lower your blood sugar!"
    
    score, fact = checker.calculate_x3_score(test_ad)
    print("-" * 30)
    print(f"Final violation score: {score}")
    print(f"Referenced regulation: {fact}")
