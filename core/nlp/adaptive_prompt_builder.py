# Adaptive Prompt Builder – Phase-1 Rule-based
from typing import Dict

class AdaptivePromptBuilder:
    def __init__(self):
        self.prompt_templates: Dict[str, str] = {}

    def add_template(self, intent: str, template: str):
        self.prompt_templates[intent.lower()] = template

    def build_prompt(self, intent: str, context: str) -> str:
        template = self.prompt_templates.get(intent.lower(), "Respond appropriately: {context}")
        return template.replace("{context}", context)

# Test
if __name__ == "__main__":
    apb = AdaptivePromptBuilder()
    apb.add_template("greeting", "Hello! How can I help you with {context}?")
    print(apb.build_prompt("greeting", "your query"))
    print(apb.build_prompt("unknown_intent", "some task"))
