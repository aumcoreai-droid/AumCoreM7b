# Adaptive Response Length – Phase-1 Rule-based
class AdaptiveResponseLength:
    def __init__(self):
        self.short_threshold = 50   # chars
        self.long_threshold = 200   # chars

    def adjust_response(self, text: str) -> str:
        length = len(text)
        if length < self.short_threshold:
            return text + "..."  # Extend short responses
        elif length > self.long_threshold:
            return text[:self.long_threshold] + "..."  # Trim long responses
        return text

# Test
if __name__ == "__main__":
    arl = AdaptiveResponseLength()
    print(arl.adjust_response("Short reply"))
    print(arl.adjust_response("This is a very long response " * 10))
