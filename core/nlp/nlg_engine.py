# Phase-1 Rule-based NLG Engine Skeleton
class NLGEngine:
    def __init__(self):
        # Basic response templates
        self.templates = {
            "greet": "Hello, how can I assist you today?",
            "bye": "Goodbye! Have a great day.",
            "thanks": "You're welcome!",
        }

    def generate(self, intent, entities=None):
        '''
        Generate a response based on intent.
        '''
        if intent in self.templates:
            return self.templates[intent]
        return "I'm still learning to generate responses for this intent."
