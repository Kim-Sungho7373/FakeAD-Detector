import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gc

class SemanticAnalyzer:
    def __init__(self):
        print("🧠 [Step 2] Loading the English Semantic Analyzer (DistilBERT)...")
        
        # Support for Apple Silicon (MPS), NVIDIA (CUDA), or CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"   -> Computation device: {self.device}")
        
        # Use DistilBERT: Lightweight, fast, and excellent for English context
        self.model_name = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # output_hidden_states=True to extract CLS vectors for similarity
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=2, 
            output_hidden_states=True
        ).to(self.device)
        self.model.eval()

        # [Vector A Space] Reference examples of English Dark Patterns & Subscription Traps
        # These represent the "DNA" of deceptive billing behavior.
        self.reference_bad_texts = [
            "Start your free trial now, $0.00 today, automatically billed full price after 7 days.",
            "No commitment, cancel anytime, but credit card required for identity verification.",
            "Special $1 offer ends in 10 minutes, recurring monthly subscription starts thereafter.",
            "Membership automatically renews at $49.99 per month unless cancelled via phone call.",
            "Hidden service fees and processing charges applied at the final checkout step."
        ]
        print("   -> Encoding global dark pattern references into vector space...")
        self.reference_embeddings = self._get_embeddings(self.reference_bad_texts)

    def clear_memory(self):
        """Prevents memory leaks during inference"""
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _get_embeddings(self, texts):
        """Converts text list into sentence embedding vectors (CLS token)"""
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token (index 0) from the last hidden layer
            sentence_embeddings = outputs.hidden_states[-1][:, 0, :] 
        return sentence_embeddings

    def calculate_x2_score(self, text):
        if not text or len(text.strip()) < 10:
            return 0.0

        print("\n🧠 [Step 2] Analyzing semantic context for deceptive intent...")
        
        # Truncate text to 512 tokens (standard for BERT-based models)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # 1. Classification Logit (Raw Probability)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            prob_deceptive = probs[0][1].item() 
            
            # 2. Vector Similarity (Comparing vs known Dark Patterns)
            current_embedding = outputs.hidden_states[-1][:, 0, :] 
            similarities = F.cosine_similarity(current_embedding, self.reference_embeddings)
            max_sim = torch.max(similarities).item()
        
        # Cleanup
        del inputs, outputs, logits, probs, current_embedding, similarities
        self.clear_memory()

        # Final X2 Score Calculation (Ensemble of Probability + Similarity)
        # Since the base model isn't fine-tuned specifically for dark patterns, 
        # Similarity (80%) is a stronger signal than raw classification (20%).
        classification_score = prob_deceptive * 100
        similarity_score = max(max_sim, 0) * 100

        x2_score = (similarity_score * 0.8) + (classification_score * 0.2)
        
        print(f"   -> 📊 Intent Detection Prob: {prob_deceptive*100:.1f}%")
        print(f"   -> 🔗 Contextual Similarity to Traps: {max_sim*100:.1f}%")
        print(f"📈 Final Semantic Risk Score: {x2_score:.2f}")
        
        return x2_score

# ==========================================
# Unit Test
# ==========================================
if __name__ == "__main__":
    analyzer = SemanticAnalyzer()
    
    # Test: A typical subscription trap phrase
    test_text = "Join now for $0! Just enter your card details for verification. You will be billed $99 after the trial."
    analyzer.calculate_x2_score(test_text)
