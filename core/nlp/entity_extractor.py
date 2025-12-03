# Phase-1 Rule-based Entity Extractor Skeleton
import re

class EntityExtractor:
    def __init__(self):
        # Define simple regex patterns for entities
        self.patterns = {
            "name": r"[A-Z][a-z]+",
            "date": r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
            "number": r"\b\d+\b"
        }

    def extract(self, text):
        entities = {}
        for entity, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                entities[entity] = matches
        return entities
