import os
import re
import torch
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FactCheckerRAG:
    def __init__(self):
        print("📚 [Step 3] Loading the Global Compliance RAG (FTC Guidelines)...")
        
        # Use a universal English model for better retrieval accuracy
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.retriever = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ Warning: OPENAI_API_KEY is not set.")
        
        self.client = OpenAI(api_key=api_key)
        
        # ⚖️ FTC Regulatory Knowledge Base (Negative Option & Dark Patterns)
        self.fact_db = [
            # 1. Disclosure Requirements
            "Material terms of the negative option offer (billing frequency, amount, cancellation) must be disclosed clearly and conspicuously before obtaining billing information.",
            "The disclosure must be placed in close proximity to the request for consent and must not be hidden in fine print or separate terms and conditions pages.",
            
            # 2. Express Informed Consent
            "Marketers must obtain the consumer's express informed consent before charging them for a subscription or recurring service.",
            "Deceptive checkboxes or 'pre-checked' boxes for additional paid services are considered unfair and deceptive acts.",
            
            # 3. Simple Cancellation (The 'Click-to-Cancel' Rule)
            "The cancellation mechanism must be at least as simple as the mechanism used to initiate the subscription (e.g., if you join online, you must be able to cancel online).",
            "Failing to provide a simple method to stop recurring charges is a direct violation of the ROSCA (Restore Online Shoppers' Confidence Act).",
            
            # 4. Misleading Pricing (Bait & Switch)
            "Promising a 'Free' or '$0' trial without disclosing that a full-price subscription begins automatically after a short period is a deceptive practice.",
            "Requiring credit card information for a 'free' service must be accompanied by a clear explanation of when and why the card will be charged."
        ]
        
        # Pre-calculate DB embeddings
        self.db_embeddings = self.retriever.encode(self.fact_db, convert_to_tensor=True)

    def calculate_x3_score(self, text):
        """Calculate a regulatory non-compliance score based on FTC guidelines via RAG."""
        if not text or len(text.strip()) < 10:
            return 0.0, "Insufficient text provided for compliance audit."

        try:
            # 1. Retrieval: Find the most relevant FTC guideline
            query_embedding = self.retriever.encode(text, convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_embedding, self.db_embeddings)[0]
            best_idx = torch.argmax(cosine_scores).item()
            retrieved_fact = self.fact_db[best_idx]
            
            # 2. Generation: GPT Audit
            prompt = f"""
            You are a senior auditor for the FTC (Federal Trade Commission) specialized in deceptive subscription practices.
            Audit the [Ad/Service Copy] based on the [Relevant Regulation].

            [Relevant Regulation]: {retrieved_fact}
            [Ad/Service Copy]: {text}

            Response Format (Strictly follow this):
            Compliance Risk: [Score from 0 to 100, where 100 is a definite violation]
            Specific Violation: [1-2 sentences explaining why it violates or follows the rule]
            Verdict: [CRITICAL / SUSPICIOUS / COMPLIANT]
            """
            
            print("   -> 🤖 GPT Auditor is cross-checking with federal regulations...")
            response = self.client.chat.completions.create(
                model="gpt-4o", # Using 4o for better reasoning in English
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            result_text = response.choices[0].message.content
            print(f"   [Audit Report]\n{result_text}")
            
            # Extract score
            score_match = re.search(r"Compliance Risk:\s*(\d+)", result_text)
            x3_score = float(score_match.group(1)) if score_match else 0.0
            
            return x3_score, retrieved_fact
            
        except Exception as e:
            print(f"⚠️ Audit Error: {e}")
            return 0.0, "An error occurred during regulatory analysis."

if __name__ == "__main__":
    checker = FactCheckerRAG()
    # Test case: Typical subscription trap language
    test_copy = "Get your FREE trial now! $0.00 today. Just enter your credit card to verify your identity."
    
    score, fact = checker.calculate_x3_score(test_copy)
    print("-" * 50)
    print(f"Final Non-Compliance Score: {score}")
    print(f"Reference FTC Guideline: {fact}")