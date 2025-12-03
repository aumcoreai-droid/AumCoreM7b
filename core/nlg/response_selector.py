# Phase-1 Rule-based Response Selector Skeleton
class ResponseSelector:
    def __init__(self):
        # Simple scoring rules
        self.priority = {
            "greet": 3,
            "thanks": 2,
            "bye": 1,
            "default": 0
        }

    def select(self, responses):
        '''
        Select best response based on priority score.
        responses: dict {intent: text}
        '''
        if not responses:
            return "No response available."
        # Sort by priority
        best_intent = max(responses.keys(), key=lambda i: self.priority.get(i, 0))
        return responses[best_intent]
