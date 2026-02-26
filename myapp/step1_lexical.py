import re
from mecab import MeCab

class LexicalAnalyzer:
    def __init__(self):
        print("✅ Mecab 형태소 분석기를 로드합니다...")
        self.mecab = MeCab()
        
        # 🚨 가중치(TF-IDF의 IDF 개념 차용)가 부여된 과대광고 사전
        # 점수가 높을수록 한 번만 등장해도 치명적인 단어입니다.
        self.lexicon = {
            "치료": 2.0, "예방": 2.0, "완치": 2.0, "항암": 2.0, "특효": 2.0,
            "100%": 1.5, "만병통치": 2.0, "기적": 1.5, "단숨에": 1.5, 
            "주문쇄도": 1.0, "단체추천": 1.0, "특수제법": 1.0, # 식약처 기만광고 적발 키워드
            "최고": 1.0, "가장 좋은": 1.0, "독소": 1.0, "부작용": 1.5,
            "체험기": 1.5, "체험사례": 1.5 # 식약처가 금지하는 후기 마케팅 키워드
        }
        
        # 🛡️ 부정어 사전 (이 단어들이 주변에 있으면 무죄 판결)
        self.negation_words = {"없", "않", "아니", "무", "안", "못"}

    def split_into_sentences(self, text):
        """텍스트를 문장 단위로 분리하여 길이 편향(Length Bias)을 방지합니다."""
        # 마침표, 느낌표, 물음표 또는 줄바꿈을 기준으로 분리
        sentences = re.split(r'[.!?\n]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 2]

    def check_negation_context(self, tokens, target_index, window_size=3):
        """
        [핵심 로직] 타겟 단어 뒤에 부정어가 오는지 문맥을 검사합니다.
        예: '부작용'(명사) + '이'(조사) + '없'(형용사) + '습니다'(어미)
        """
        # 타겟 단어 뒤의 window_size 만큼의 형태소를 살펴봅니다.
        end_index = min(target_index + window_size + 1, len(tokens))
        context_tokens = tokens[target_index + 1 : end_index]
        
        for word, pos in context_tokens:
            # VA(형용사-없다), VX(보조용언-않다), MAG(부사-안,못) 등을 체크
            if word in self.negation_words or pos in ['VA', 'VX', 'MAG']:
                return True # 부정어가 존재함!
        return False

    def calculate_x1_score(self, text):
        if not text:
            return 0.0

        print("\n🔍 [Step 1] 텍스트 과장도(Lexical Score) 분석 시작...")
        
        sentences = self.split_into_sentences(text)
        total_sentences = len(sentences)
        
        if total_sentences == 0:
            return 0.0

        total_penalty = 0.0
        detected_issues = []

        # 문장 단위로 검사하여 텍스트가 길어져도 점수가 희석되지 않게 방어
        for sentence in sentences:
            tokens = self.mecab.pos(sentence)
            sentence_flagged = False
            
            for i, (word, pos) in enumerate(tokens):
                # 사전에 있는 금칙어인지 확인
                if word in self.lexicon:
                    # 부정어 문맥 체크 ("부작용이 없습니다" 필터링)
                    is_negated = self.check_negation_context(tokens, i)
                    
                    if is_negated:
                        detected_issues.append(f"🛡️ 무죄(부정어 동반): '{word}' (문장: {sentence})")
                    else:
                        weight = self.lexicon[word]
                        total_penalty += weight
                        sentence_flagged = True
                        detected_issues.append(f"🚨 적발: '{word}' (가중치: +{weight})")
            
            # 한 문장에 금칙어가 여러 번 나와도 1차원적으로 폭발하지 않도록 패널티 상한선 부여
            if sentence_flagged:
                total_penalty += 0.5 # 문장 자체의 불량도 추가 점수

        # 🧮 X1 스코어 계산 (0 ~ 100점 스케일링)
        # 공식: (총 패널티 / 전체 문장 수)를 기준으로 점수화하되, 로그 스케일 등을 써서 100점 상한선 적용
        raw_score = (total_penalty / total_sentences) * 50
        x1_score = min(raw_score, 100.0)

        # 결과 리포트 출력
        print(f"\n[분석 리포트]")
        print(f" - 전체 문장 수: {total_sentences}문장")
        for issue in detected_issues:
            print(f" {issue}")
            
        print(f"📈 최종 X1 점수: {x1_score:.2f} / 100.0 점")
        return x1_score

# ==========================================
# 실제 실행 테스트 코드
# ==========================================
if __name__ == "__main__":
    analyzer = LexicalAnalyzer()
    
    print("==================================================")
    print("테스트 1: 부정어가 포함된 정상적인 광고 (오탐 방지 테스트)")
    test_text_1 = "이 제품은 식약처 인증을 받았습니다. 피부 트러블이나 부작용이 전혀 없습니다. 안심하고 사용하세요."
    analyzer.calculate_x1_score(test_text_1)
    
    print("\n==================================================")
    print("테스트 2: 극단적인 과대광고 (허위 탐지 테스트)")
    test_text_2 = "단 일주일 만에 지방이 100% 분해되는 기적을 경험하세요! 이것은 암도 완치하는 만병통치 약입니다. 무조건 구매하세요. 최고의 선택입니다."
    analyzer.calculate_x1_score(test_text_2)