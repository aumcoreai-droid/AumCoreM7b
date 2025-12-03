# Phase-1 Rule-based Dialog Manager Skeleton
class DialogManager:
    def __init__(self):
        # Track conversation state
        self.state = {}

    def update_state(self, intent, entities=None):
        self.state['intent'] = intent
        self.state['entities'] = entities

    def decide_action(self):
        # Rule-based decision
        if self.state.get('intent') == "greet":
            return "greet"
        elif self.state.get('intent') == "bye":
            return "bye"
        elif self.state.get('intent') == "thanks":
            return "thanks"
        else:
            return "default"
