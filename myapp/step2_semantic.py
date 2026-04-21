import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gc

class SemanticAnalyzer:
    def __init__(self):
        print("🧠 [Step 2] Loading the semantic deep-learning analyzer (KoELECTRA)...")
        
        # M1 Max GPU(MPS) 세팅
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # 가볍고 성능이 뛰어난 KoELECTRA 모델 로드
        self.model_name = "monologg/koelectra-base-v3-discriminator"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 🌟 핵심: output_hidden_states=True 옵션을 켜야 
        # 분류 Logit과 코사인 유사도용 Vector를 한 번의 연산으로 둘 다 뽑아낼 수 있습니다!
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2, # 0: 정상, 1: 과대광고
            output_hidden_states=True
        ).to(self.device)
        self.model.eval() # 추론 모드 전환

        # [벡터 A 구축] 기존에 적발된 허위광고 레퍼런스 문장들 (예시)
        self.reference_bad_texts = [
            "단 일주일 만에 지방이 100% 분해되는 기적의 크림",
            "식약처에서 인증한 만병통치약, 암세포 완벽 제거",
            "이것만 먹으면 독소가 배출되고 세포가 즉각 재생됩니다",
            "의사들이 무조건 추천하는 부작용 없는 완치제"
        ]
        print("   -> Encoding reference misleading-ad sentences into vector space...")
        self.reference_embeddings = self._get_embeddings(self.reference_bad_texts)

    def clear_memory(self):
        """메모리 누수 방지용 가비지 컬렉션"""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def _get_embeddings(self, texts):
        """텍스트 리스트를 입력받아 문장 임베딩 벡터를 반환합니다."""
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 마지막 레이어의 hidden states에서 [CLS] 토큰(인덱스 0)의 벡터를 문장 대표 벡터로 사용
            sentence_embeddings = outputs.hidden_states[-1][:, 0, :] 
        return sentence_embeddings

    def calculate_x2_score(self, text):
        if not text or len(text.strip()) < 5:
            return 0.0

        print("\n🧠 [Step 2] Starting semantic context and similarity analysis...")
        
        # 너무 긴 텍스트는 PLM 한계(512토큰)에 걸리므로 잘라서 입력
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad(): # 역전파(학습) 연산을 꺼서 M1 메모리 절약
            outputs = self.model(**inputs)
            
            # =========================================================
            # [수식 1] 분류 확률 (Softmax) : X2
            # =========================================================
            logits = outputs.logits # 모델의 날것 출력값 (z)
            probs = F.softmax(logits, dim=-1) # Softmax 적용: e^z / sum(e^z)
            
            # 라벨 1(과대광고)에 해당하는 확률값 (0.0 ~ 1.0)
            prob_fake = probs[0][1].item() 
            
            # =========================================================
            # [수식 2] 의미론적 유사도 (Cosine Similarity)
            # =========================================================
            # 현재 검사 중인 텍스트의 벡터 B 추출
            current_embedding = outputs.hidden_states[-1][:, 0, :] 
            
            # 레퍼런스 벡터 A들과 벡터 B의 유사도 계산: (A * B) / (|A| * |B|)
            similarities = F.cosine_similarity(current_embedding, self.reference_embeddings)
            
            # 가장 문맥이 비슷하다고 판정된 레퍼런스와의 최고 유사도 점수
            max_sim = torch.max(similarities).item()
        
        # 분석이 끝나면 즉시 VRAM 반환
        del inputs, outputs, logits, probs, current_embedding, similarities
        self.clear_memory()

        # =========================================================
        # 최종 X2 스코어 산출 (분류 확률 + 유사도 앙상블)
        # =========================================================
        # 분류기 확률을 100점 만점으로 변환
        classification_score = prob_fake * 100
        # 코사인 유사도를 100점 만점으로 변환 (유사도가 0 이하면 0점 처리)
        similarity_score = max(max_sim, 0) * 100
        
        # 💡 [실무 팁] 혼합 가중치 적용
        # 현재 KoELECTRA 모델은 '과대광고 전용'으로 파인튜닝 되지 않은 쌩얼 상태입니다.
        # 따라서 Softmax 분류 확률은 랜덤에 가깝고, 코사인 유사도가 훨씬 정확합니다.
        # 유사도 점수에 80%, 분류 점수에 20% 가중치를 주어 최종 점수를 만듭니다.
        x2_score = (similarity_score * 0.8) + (classification_score * 0.2)
        
        print(f"   -> 📊 Model classification probability (Softmax): {prob_fake*100:.1f}%")
        print(f"   -> 🔗 Maximum contextual similarity (Cosine Sim): {max_sim*100:.1f}%")
        print(f"📈 Final X2 score (0-100): {x2_score:.2f}")
        
        return x2_score

# ==========================================
# 단독 테스트 코드
# ==========================================
if __name__ == "__main__":
    analyzer = SemanticAnalyzer()
    
    # 레퍼런스("단 일주일 만에...")와 의미가 유사한 문장 테스트
    test_text = "단 7일만 투자하세요! 지방이 완전히 파괴되는 놀라운 마법을 겪게 됩니다."
    analyzer.calculate_x2_score(test_text)
