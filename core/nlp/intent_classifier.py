# Phase-1 Rule-based Intent Classifier Skeleton
class IntentClassifier:
    def __init__(self):
        # Simple keyword-based rules
        self.rules = {
            "greet": ["hello", "hi", "hey"],
            "bye": ["bye", "goodbye", "see you"],
            "thanks": ["thanks", "thank you"],
        }

    def classify(self, text):
        text = text.lower()
        for intent, keywords in self.rules.items():
            for kw in keywords:
                if kw in text:
                    return intent
        return "unknown"
