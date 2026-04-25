import numpy as np
import xgboost as xgb
import shap

class XAIScorer:
    def __init__(self):
        print("📊 [Step 4] Loading the XAI Scoring Engine (XGBoost + SHAP)...")
        
        # 1. Generating synthetic data for PoC training
        # Features: X1 (Lexical Bait), X2 (Semantic Intent), X3 (Regulatory Violation)
        np.random.seed(42)
        X_train = np.random.rand(1000, 3) * 100 
        
        # Hypothetical labeling logic: 
        # In subscription traps, Regulatory non-compliance (X3) and Semantic intent (X2) 
        # carry the most weight in determining a 'Deceptive' label.
        # Target: 1 (Deceptive/Trap), 0 (Compliant)
        y_train = ((X_train[:, 0]*0.2 + X_train[:, 1]*0.4 + X_train[:, 2]*0.4) > 50).astype(int)

        # 2. Define and train the XGBoost Classifier
        # This model acts as the 'decision-making brain' of the auditor.
        self.model = xgb.XGBClassifier(
            n_estimators=50, 
            max_depth=3, 
            learning_rate=0.1, 
            eval_metric='logloss',
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # 3. Initialize the SHAP TreeExplainer
        # This allows us to see exactly how each feature influenced the final risk score.
        self.explainer = shap.TreeExplainer(self.model)
        print("   -> 🧠 Machine-learning scoring engine is ready (Global Version).")

    def calculate_final_score_and_explain(self, x1, x2, x3):
        """
        Calculates the probability of being a Dark Pattern and provides an XAI trace.
        x1: Lexical Score, x2: Semantic Score, x3: RAG Compliance Score
        """
        # Convert input scores into a numpy array for the model
        X_input = np.array([[x1, x2, x3]])
        
        # =========================================================
        # 1. Final Risk Scoring
        # Predict_proba returns the probability of class 1 (Deceptive)
        # =========================================================
        probabilities = self.model.predict_proba(X_input)
        final_score = probabilities[0][1] * 100  # Scale to 0-100 range
        
        # =========================================================
        # 2. XAI (SHAP Value Calculation)
        # SHAP explains the deviation from the base value (average prediction).
        # =========================================================
        shap_values = self.explainer.shap_values(X_input)
        
        # Handle different SHAP output formats (list or array)
        if isinstance(shap_values, list):
            # For binary classification, we take the values for the positive class (1)
            shap_vals = shap_values[1][0] 
        elif len(shap_values.shape) == 3: # Some versions return (obs, class, features)
            shap_vals = shap_values[0][1]
        else:
            shap_vals = shap_values[0]

        # Extract the Base Value (The average risk score of the training set)
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            base_value = self.explainer.expected_value[1]
        else:
            base_value = self.explainer.expected_value

        # Note: final_score ≈ (base_value + sum(shap_vals)) * 100
        return final_score, shap_vals, base_value

if __name__ == "__main__":
    scorer = XAIScorer()
    # Example: A page with some lexical bait but high semantic deception and regulatory failure
    f_score, s_vals, b_val = scorer.calculate_final_score_and_explain(30.0, 75.0, 85.0)
    
    print("\n" + "="*50)
    print(f"Audit Result: {f_score:.2f}% Deception Probability")
    print("-" * 50)
    print(f"Lexical Impact: {s_vals[0]:+.2f}")
    print(f"Semantic Impact: {s_vals[1]:+.2f}")
    print(f"Regulatory Impact: {s_vals[2]:+.2f}")
    print("="*50)