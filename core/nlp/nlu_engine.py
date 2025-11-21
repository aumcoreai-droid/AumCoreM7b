import re

class NLUEngine:
    def __init__(self):
        self.intents = {
            'greet': [r'hello', r'hi', r'hey'],
            'bye': [r'bye', r'see you', r'goodbye'],
            'order_pizza': [r'order.*pizza', r'pizza', r'want.*pizza'],
            'weather': [r'weather']
        }

    def detect_intent(self, text):
        if not text.strip():
            return {'intent': 'unknown', 'confidence': 0.0, 'matches': [], 'text': text}

        best_intent = 'unknown'
        best_matches = []
        best_conf = 0.0

        for intent, patterns in self.intents.items():
            matches = []
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    matches.append(pat)
            if matches and len(matches) > best_conf:
                best_conf = float(len(matches))
                best_intent = intent
                best_matches = matches

        if best_conf > 0:
            confidence = 1.0
        else:
            confidence = 0.0

        return {
            'intent': best_intent,
            'confidence': confidence,
            'matches': best_matches,
            'text': text
        }

if __name__ == '__main__':
    engine = NLUEngine()
    txt = input("Enter text: ")
    result = engine.detect_intent(txt)
    print("Detected:", result)
