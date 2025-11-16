
import os
import json

class VectorStore:
    def __init__(self):
        self.documents = []
        self.metadata = []

    def add(self, doc_id, text, metadata=None):
        self.documents.append(text)
        self.metadata.append(metadata or {})

    def query(self, text, k=5):
        results = []
        for i, doc in enumerate(self.documents):
            if text.lower() in doc.lower():
                results.append({
                    "document": doc,
                    "metadata": self.metadata[i],
                    "similarity": 0.8
                })
            if len(results) >= k:
                break
        return results

vector_store = VectorStore()
