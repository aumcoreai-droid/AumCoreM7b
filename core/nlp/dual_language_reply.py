# core/nlp/dual_language_reply.py - AumCore AI Dual Language Engine
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Dict, List, Any, Tuple
import json
import time
import gc

class DualLanguageEngine:
    """Mistral-7B based Dual Language (Hindi-English) Response Engine"""
    
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
            self.logger.info(f"Loading Dual Language model: {self.model_name}")
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
            self.logger.info("✅ Dual Language model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def detect_input_language(self, text: str) -> str:
        """Detect if input is Hindi or English"""
        hindi_chars = any('\u0900' <= char <= '\u097F' for char in text)
        return "hi" if hindi_chars else "en"
    
    def generate_bilingual_response(self, user_input: str, target_language: str = "both") -> Dict[str, str]:
        """Generate response in both Hindi and English"""
        if not self.is_loaded:
            if not self.load_model():
                return {"error": "Model loading failed"}
        
        input_lang = self.detect_input_language(user_input)
        
        try:
            if target_language == "both":
                # Generate both Hindi and English responses
                english_prompt = f"Provide a helpful response in English: {user_input}\nResponse:"
                hindi_prompt = f"हिंदी में उपयोगी जवाब दें: {user_input}\nजवाब:"
                
                english_response = self._generate_single(english_prompt)
                hindi_response = self._generate_single(hindi_prompt)
                
                return {
                    "input_language": input_lang,
                    "english": english_response,
                    "hindi": hindi_response,
                    "mixed": f"{english_response}\n\n{hindi_response}"
                }
            
            elif target_language == "en":
                prompt = f"Provide response in English: {user_input}\nResponse:"
                response = self._generate_single(prompt)
                return {"english": response, "input_language": input_lang}
            
            elif target_language == "hi":
                prompt = f"हिंदी में जवाब दें: {user_input}\nजवाब:"
                response = self._generate_single(prompt)
                return {"hindi": response, "input_language": input_lang}
            
            else:
                return {"error": "Invalid target language"}
                
        except Exception as e:
            self.logger.error(f"Bilingual generation error: {e}")
            return {"error": str(e)}
    
    def _generate_single(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate single response"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            device = self.model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            self._optimize_memory()
            return response
            
        except Exception as e:
            self.logger.error(f"Single generation error: {e}")
            return "Sorry, I couldn't generate a response."
    
    def translate_text(self, text: str, target_lang: str) -> str:
        """Translate text between Hindi and English"""
        if target_lang == "hi":
            prompt = f"Translate to Hindi: {text}\nTranslation:"
        elif target_lang == "en":
            prompt = f"Translate to English: {text}\nTranslation:"
        else:
            return "Unsupported language"
        
        return self._generate_single(prompt, max_tokens=80)
    
    def get_language_preference(self, user_history: List[str]) -> str:
        """Analyze user history to determine language preference"""
        if not user_history:
            return "en"
        
        hindi_count = sum(1 for msg in user_history if self.detect_input_language(msg) == "hi")
        english_count = len(user_history) - hindi_count
        
        if hindi_count > english_count:
            return "hi"
        elif english_count > hindi_count:
            return "en"
        else:
            return "both"

if __name__ == "__main__":
    print("🚀 Testing Dual Language Engine...")
    engine = DualLanguageEngine()
    
    test_inputs = [
        "What is artificial intelligence?",
        "कृत्रिम बुद्धिमत्ता क्या है?",
        "Hello, how are you today?",
        "नमस्ते, आप कैसे हैं?"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n🧪 Test {i}: '{user_input}'")
        start_time = time.time()
        result = engine.generate_bilingual_response(user_input, "both")
        end_time = time.time()
        
        print(f"🌍 Input Language: {result.get('input_language', 'unknown')}")
        if 'english' in result:
            print(f"🇺🇸 English: {result['english']}")
        if 'hindi' in result:
            print(f"🇮🇳 Hindi: {result['hindi']}")
        print(f"⏱️ Time: {end_time - start_time:.2f}s")
        time.sleep(2)
    
    print("\n✅ Dual Language Engine testing completed!")
