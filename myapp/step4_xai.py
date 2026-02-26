import numpy as np
import xgboost as xgb
import shap

class XAIScorer:
    def __init__(self):
        print("📊 [Step 4] XGBoost 앙상블 모델 및 SHAP 설명기(XAI) 로드 중...")
        
        # 1. PoC용 가상 데이터 생성 (실무에서는 실제 라벨링된 DB를 불러옵니다)
        # X1(단어), X2(문맥), X3(팩트체크) 점수를 랜덤 생성
        np.random.seed(42)
        X_train = np.random.rand(1000, 3) * 100 
        
        # 가상의 정답(Label) 생성 로직: X2와 X3가 높을수록 과대광고(1)일 확률이 높음
        # y = 1 (과대광고), y = 0 (정상)
        y_train = ((X_train[:, 0]*0.2 + X_train[:, 1]*0.4 + X_train[:, 2]*0.4) > 50).astype(int)

        # 2. XGBoost 모델 정의 및 학습 (Logistic 변환 내장)
        self.model = xgb.XGBClassifier(
            n_estimators=50, 
            max_depth=3, 
            learning_rate=0.1, 
            eval_metric='logloss',
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # 3. SHAP TreeExplainer 초기화 (XAI)
        self.explainer = shap.TreeExplainer(self.model)
        print("   -> 🧠 머신러닝 스코어링 엔진 세팅 완료!")

    def calculate_final_score_and_explain(self, x1, x2, x3):
        # 입력값을 numpy 배열로 변환
        X_input = np.array([[x1, x2, x3]])
        
        # =========================================================
        # 1. 최종 스코어링 (Logistic/Sigmoid 변환)
        # XGBoost의 predict_proba는 내부적으로 Z값에 Sigmoid를 씌워 0~1 확률을 반환합니다.
        # =========================================================
        probabilities = self.model.predict_proba(X_input)
        final_score = probabilities[0][1] * 100  # 클래스 1(과대광고)일 확률을 100점 만점으로 변환
        
        # =========================================================
        # 2. XAI (SHAP Value 계산)
        # =========================================================
        shap_values = self.explainer.shap_values(X_input)
        
        # 이진 분류의 경우 버전/세팅에 따라 리스트로 나올 수 있으므로 안전하게 추출
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0] 
        else:
            shap_vals = shap_values[0]

        # Base Value (모델의 평균 예측값/절편)
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            base_value = self.explainer.expected_value[1]
        else:
            base_value = self.explainer.expected_value

        return final_score, shap_vals, base_value