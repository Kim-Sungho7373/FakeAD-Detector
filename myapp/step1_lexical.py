import re

class LexicalAnalyzer:
    def __init__(self):
        print("✅ Loading the English Dark Pattern Lexical Analyzer...")
        
        # 🚨 Subscription Trap & Dark Pattern Lexicon (English)
        # Weights indicate the level of risk associated with "Bait-and-Switch" tactics.
        self.lexicon = {
            # Bait Triggers
            "free trial": 2.5, 
            "0.00": 1.5,
            "for free": 2.0,
            "get started free": 1.5,
            "7 days free": 2.0,
            "30 days free": 2.0,
            
            # Hidden Continuity / Trap Triggers
            "auto-renew": 3.0,
            "automatic renewal": 3.0,
            "recurring billing": 3.0,
            "monthly subscription": 2.0,
            "billed annually": 1.5,
            "hidden fee": 3.0,
            "card required": 2.5,
            "credit card required": 2.5,
            
            # Forced Urgency / Scarcity (Dark Patterns)
            "limited time offer": 1.0,
            "exclusive deal": 1.0,
            "ends soon": 1.0,
            "cancel anytime": 1.0 # Often used to mask the difficulty of actual cancellation
        }
        
        # 🛡️ Negation words (to prevent false positives like "No hidden fees")
        self.negation_words = {"no", "not", "never", "without", "zero", "none"}

    def split_into_sentences(self, text):
        """Splits English text into sentences using punctuation."""
        sentences = re.split(r'[.!?\n]+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def check_negation_context(self, sentence, target_phrase):
        """
        Checks if a negation word appears shortly before the target phrase.
        Example: "No [hidden fees]" -> Cleared.
        """
        sentence = sentence.lower()
        target_phrase = target_phrase.lower()
        
        # Look for words within a 3-word window before the phrase
        words = sentence.split()
        try:
            # Find the starting index of the target phrase in terms of word position
            phrase_start_index = sentence.find(target_phrase)
            prefix_text = sentence[:phrase_start_index].split()
            
            # Check the last 3 words before the phrase
            lookback_window = prefix_text[-3:] if len(prefix_text) >= 3 else prefix_text
            
            for neg in self.negation_words:
                if neg in lookback_window:
                    return True
            return False
        except:
            return False

    def calculate_x1_score(self, text):
        if not text:
            return 0.0

        print("\n🔍 [Step 1] Analyzing text for 'Bait-and-Switch' lexical triggers...")
        
        text_lower = text.lower()
        sentences = self.split_into_sentences(text)
        total_sentences = len(sentences)
        
        if total_sentences == 0:
            return 0.0

        total_penalty = 0.0
        detected_issues = []

        # Check for each phrase in the lexicon
        for phrase, weight in self.lexicon.items():
            # Find all occurrences of the phrase
            matches = re.finditer(re.escape(phrase), text_lower)
            
            for match in matches:
                # Find which sentence contains this match to check context
                start_pos = match.start()
                # Find the sentence containing this position
                current_sentence = next((s for s in sentences if phrase in s.lower()), "")
                
                is_negated = self.check_negation_context(current_sentence, phrase)
                
                if is_negated:
                    detected_issues.append(f"🛡️ Cleared by negation: '{phrase}' in '{current_sentence[:40]}...'")
                else:
                    total_penalty += weight
                    detected_issues.append(f"🚨 Flagged: '{phrase}' (weight: +{weight})")

        # 🧮 X1 Score Scaling (0 ~ 100)
        # We use a density-based scoring approach
        raw_score = (total_penalty / (total_sentences * 0.5)) * 20
        x1_score = min(raw_score, 100.0)

        # Report logs
        print(f"[Lexical Report] Penalty: {total_penalty:.1f}, Sentences: {total_sentences}")
        print(f"📈 Final Lexical Risk Score: {x1_score:.2f} / 100.0")
        
        return x1_score

# ==========================================
# Local Test Logic
# ==========================================
if __name__ == "__main__":
    analyzer = LexicalAnalyzer()
    
    print("\n" + "="*50)
    print("Test 1: Safe ad with clear disclosures")
    test_1 = "Get our service for $10 a month. No hidden fees. Cancel anytime."
    analyzer.calculate_x1_score(test_1)
    
    print("\n" + "="*50)
    print("Test 2: High-risk subscription trap")
    test_2 = "Start your 7 days free trial now! Only $0.00 today. Automatic renewal applies after the trial period. Credit card required for verification."
    analyzer.calculate_x1_score(test_2)
    