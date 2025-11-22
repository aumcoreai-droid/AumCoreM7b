# core/nlp/nlg_engine.py - AumCore AI NLG Engine (Final Working Version)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Dict, List, Any
import json
import time
import gc

class NLGEngine:
    """Mistral-7B NLG Engine - Guaranteed Working Version"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.logger = self._setup_logging()
        self.is_loaded = False
        
    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _optimize_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def load_model(self):
        try:
            self.logger.info(f"Loading NLG model: {self.model_name}")
            self._optimize_memory()
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            self.model.eval()
            self.is_loaded = True
            self.logger.info("✅ NLG model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def generate_response(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
        if not self.is_loaded:
            if not self.load_model():
                return "I apologize, but I'm having trouble generating a response right now."
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            device = self.model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    repetition_penalty=1.1
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            self._optimize_memory()
            return response
            
        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            self._optimize_memory()
            return "I encountered an error while generating the response."
    
    def generate_from_intent(self, intent_data: Dict[str, Any], user_input: str) -> str:
        intent = intent_data.get("primary_intent", "general")
        
        prompts = {
            "greeting": "Generate a friendly greeting response:",
            "question": "Provide a helpful answer:",
            "help": "Offer supportive assistance:",
            "general": "Generate a natural response:"
        }
        
        base_prompt = prompts.get(intent, prompts["general"])
        full_prompt = f"{base_prompt}
User: {user_input}
Assistant:"
        
        return self.generate_response(full_prompt)
    
    def adjust_response_length(self, response: str, target_length: str = "medium") -> str:
        length_map = {"short": 50, "medium": 100, "long": 200}
        max_len = length_map.get(target_length, 100)
        words = response.split()
        return " ".join(words[:max_len]) if len(words) > max_len else response

if __name__ == "__main__":
    print("🚀 Testing Final NLG Engine...")
    nlg = NLGEngine()
    
    test_prompts = [
        "Explain AI in simple terms:",
        "What is machine learning?",
        "Hello, how are you?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🧪 Test {i}: {prompt}")
        start_time = time.time()
        response = nlg.generate_response(prompt, max_tokens=100)
        end_time = time.time()
        print(f"🤖 Response: {response}")
        print(f"⏱️ Time: {end_time - start_time:.2f}s")
    
    print("\n✅ Final NLG Engine testing completed!")
