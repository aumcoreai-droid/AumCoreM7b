# core/nlp/nlu_engine.py - AumCore AI NLU Engine
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Dict, List, Any
import json
import time
import gc

class NLUEngine:
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
            self.logger.info(f"Loading NLU model: {self.model_name}")
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
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def extract_intent(self, text: str) -> Dict[str, Any]:
        if not self.is_loaded:
            if not self.load_model():
                return self._rule_based_intent(text)
        
        try:
            prompt = f'Analyze: "{text}". Return JSON: {{"intent": "category", "confidence": 0.8}}'
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            device = self.model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "{" in response:
                json_str = response.split("{", 1)[-1]
                json_str = "{" + json_str.split("}")[0] + "}"
                intent_data = json.loads(json_str)
                return {
                    "primary_intent": intent_data.get("intent", "general"),
                    "confidence": float(intent_data.get("confidence", 0.7)),
                    "entities": [],
                    "raw_text": text,
                    "method": "model"
                }
        except Exception as e:
            self.logger.warning(f"Model intent failed: {e}")
        
        return self._rule_based_intent(text)
    
    def _rule_based_intent(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        if any(word in text_lower for word in ["hello", "hi", "hey"]):
            intent, confidence = "greeting", 0.9
        elif "?" in text:
            intent, confidence = "question", 0.8
        elif any(word in text_lower for word in ["help", "support", "problem"]):
            intent, confidence = "help", 0.85
        else:
            intent, confidence = "statement", 0.7
        
        return {
            "primary_intent": intent,
            "confidence": confidence,
            "entities": [],
            "raw_text": text,
            "method": "rule_based"
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        if not self.is_loaded:
            if not self.load_model():
                return self._rule_based_sentiment(text)
        
        try:
            prompt = f'Sentiment: "{text}" -> {{"positive": 0.7, "negative": 0.1, "neutral": 0.2}}'
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=200, truncation=True)
            device = self.model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=30,
                    temperature=0.2,
                    do_sample=True
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "{" in response:
                json_str = response.split("{", 1)[-1]
                json_str = "{" + json_str.split("}")[0] + "}"
                return json.loads(json_str)
        except Exception as e:
            self.logger.warning(f"Model sentiment failed: {e}")
        
        return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        positive_words = ["good", "great", "excellent", "happy", "love", "amazing"]
        negative_words = ["bad", "terrible", "hate", "angry", "sad", "problem"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        total = positive_count + negative_count + 1
        
        return {
            "positive": round(positive_count / total, 2),
            "negative": round(negative_count / total, 2),
            "neutral": round(1 / total, 2)
        }
    
    def detect_language(self, text: str) -> str:
        if any('\u0900' <= char <= '\u097F' for char in text):
            return "hi"
        elif any('\u0600' <= char <= '\u06FF' for char in text):
            return "ar"
        else:
            return "en"
    
    def process_query(self, text: str) -> Dict[str, Any]:
        processed_text = text.strip()
        if not processed_text.endswith(('?', '!', '.')):
            processed_text += '.'
        
        if not self.is_loaded:
            self.load_model()
        
        try:
            intent_result = self.extract_intent(processed_text)
            time.sleep(0.2)
            sentiment_result = self.analyze_sentiment(processed_text)
            
            result = {
                "intent": intent_result,
                "sentiment": sentiment_result,
                "language": self.detect_language(processed_text),
                "processed_text": processed_text,
                "timestamp": int(time.time()),
                "model_loaded": self.is_loaded,
                "gpu_used": torch.cuda.is_available()
            }
            
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            result = {
                "intent": self._rule_based_intent(processed_text),
                "sentiment": self._rule_based_sentiment(processed_text),
                "language": self.detect_language(processed_text),
                "processed_text": processed_text,
                "timestamp": int(time.time()),
                "model_loaded": False,
                "gpu_used": False
            }
        
        self._optimize_memory()
        return result

if __name__ == "__main__":
    nlu = NLUEngine()
    result = nlu.process_query("Hello, how are you?")
    print("NLU Result:", json.dumps(result, indent=2))
