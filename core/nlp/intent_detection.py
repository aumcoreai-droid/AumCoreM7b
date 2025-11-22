# core/nlp/intent_detection.py - AumCore AI Intent Detection Engine
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import Dict, List, Any
import json
import time
import gc

class IntentDetectionEngine:
    """Mistral-7B based Intent Detection Engine"""
    
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
            self.logger.info(f"Loading Intent Detection model: {self.model_name}")
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
            self.logger.info("✅ Intent Detection model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def detect_intent(self, text: str) -> Dict[str, Any]:
        """Detect intent from user input"""
        if not self.is_loaded:
            if not self.load_model():
                return self._rule_based_intent(text)
        
        try:
            prompt = f"""Analyze the user message and classify its intent. Return ONLY JSON format:
            {{"primary_intent": "intent_category", "confidence": 0.95, "entities": [], "sub_intents": []}}

            User: "{text}"

            Analysis: {{"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=300, truncation=True)
            device = self.model.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    max_new_tokens=80,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract JSON from response
            if "{" in response:
                json_str = response.split("{", 1)[-1]
                json_str = "{" + json_str.split("}")[0] + "}"
                intent_data = json.loads(json_str)
                
                return {
                    "primary_intent": intent_data.get("primary_intent", "unknown"),
                    "confidence": float(intent_data.get("confidence", 0.5)),
                    "entities": intent_data.get("entities", []),
                    "sub_intents": intent_data.get("sub_intents", []),
                    "method": "model"
                }
            else:
                return self._rule_based_intent(text)
                
        except Exception as e:
            self.logger.error(f"Intent detection error: {e}")
            return self._rule_based_intent(text)
    
    def _rule_based_intent(self, text: str) -> Dict[str, Any]:
        """Rule-based intent detection as fallback"""
        text_lower = text.lower()
        
        # Intent patterns with confidence scores
        intent_patterns = {
            "greeting": {
                "patterns": ["hello", "hi", "hey", "namaste", "good morning", "good afternoon"],
                "confidence": 0.9
            },
            "farewell": {
                "patterns": ["bye", "goodbye", "see you", "take care", "good night"],
                "confidence": 0.9
            },
            "question": {
                "patterns": ["what", "when", "where", "why", "how", "who", "which", "?"],
                "confidence": 0.8
            },
            "help_request": {
                "patterns": ["help", "support", "problem", "issue", "trouble", "assist"],
                "confidence": 0.85
            },
            "information_request": {
                "patterns": ["tell me", "explain", "what is", "information", "details"],
                "confidence": 0.8
            },
            "complaint": {
                "patterns": ["complaint", "bad", "terrible", "awful", "hate", "disappointed", "angry"],
                "confidence": 0.9
            },
            "gratitude": {
                "patterns": ["thank", "thanks", "appreciate", "grateful"],
                "confidence": 0.95
            },
            "request": {
                "patterns": ["can you", "could you", "please", "would you", "I need", "I want"],
                "confidence": 0.8
            }
        }
        
        # Find matching intent
        matched_intent = "general"
        confidence = 0.6
        entities = []
        sub_intents = []
        
        for intent, data in intent_patterns.items():
            for pattern in data["patterns"]:
                if pattern in text_lower:
                    matched_intent = intent
                    confidence = data["confidence"]
                    break
        
        # Extract simple entities
        if "weather" in text_lower:
            entities.append({"type": "topic", "value": "weather"})
        if any(word in text_lower for word in ["time", "clock"]):
            entities.append({"type": "topic", "value": "time"})
        
        return {
            "primary_intent": matched_intent,
            "confidence": confidence,
            "entities": entities,
            "sub_intents": sub_intents,
            "method": "rule_based"
        }
    
    def batch_detect_intent(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Detect intents for multiple texts"""
        results = []
        for text in texts:
            results.append(self.detect_intent(text))
            time.sleep(0.1)  # Prevent rapid processing
        return results
    
    def get_intent_categories(self) -> Dict[str, List[str]]:
        """Get available intent categories"""
        return {
            "communication": ["greeting", "farewell", "gratitude"],
            "information": ["question", "information_request"], 
            "assistance": ["help_request", "request"],
            "feedback": ["complaint", "praise"],
            "general": ["general", "unknown"]
        }

if __name__ == "__main__":
    print("🚀 Testing Intent Detection Engine...")
    engine = IntentDetectionEngine()
    
    test_messages = [
        "Hello, how are you today?",
        "What is the weather like in Delhi?",
        "I need help with my account login",
        "Thank you for your assistance!",
        "This product is not working properly",
        "Can you tell me about machine learning?",
        "Goodbye, see you tomorrow!"
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n🧪 Test {i}: '{message}'")
        start_time = time.time()
        result = engine.detect_intent(message)
        end_time = time.time()
        
        print(f"🎯 Intent: {result['primary_intent']}")
        print(f"📊 Confidence: {result['confidence']:.2f}")
        print(f"🔧 Method: {result['method']}")
        if result['entities']:
            print(f"🏷️ Entities: {result['entities']}")
        print(f"⏱️ Time: {end_time - start_time:.2f}s")
        time.sleep(1)
    
    print("\n✅ Intent Detection Engine testing completed!")
