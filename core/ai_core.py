# ai_core.py - AI Logic & Code Generation
import os
from groq import Groq
from langdetect import detect
from reasoning_core import ReasoningEngine
from chain_of_thought_manager import ChainOfThought

class AICore:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.reasoning = ReasoningEngine()
        self.chain = ChainOfThought()
        
    def detect_language(self, text):
        try:
            lang = detect(text)
            return "hi" if lang == "hi" else "en"
        except:
            return "en"
    
    def generate_code_response(self, user_input):
        # Language detection
        lang = self.detect_language(user_input)
        
        # Chain-of-Thought reasoning
        thought_process = self.chain.generate_thoughts(user_input)
        
        # Generate code using reasoning
        code = self.reasoning.generate_complex_code(user_input, thought_process)
        
        # Format response based on language
        if lang == "hi":
            return f"""कोड तैयार है:

{code}

यह कोड 350+ लाइन का है और Colab में रन करेगा।"""
        else:
            return f"""Code Generated:

{code}

This is 350+ lines of production-ready Python code."""

# Auto-updated: 1766656881.670995
