"""
Advanced Language Detection Module for AumCore AI
Version: 2.0.1
Author: AumCore AI
"""

import re
from typing import Optional, Tuple, Dict
import langdetect
from langdetect import DetectorFactory, lang_detect_exception

# Ensure consistent results
DetectorFactory.seed = 0

class LanguageDetector:
    """Professional language detection with multi-layered approach"""
    
    # Language scripts detection ranges
    SCRIPT_RANGES = {
        'hi': [(0x0900, 0x097F)],  # Devanagari
        'en': [(0x0041, 0x007A)],  # Basic Latin
        'es': [(0x0041, 0x007A)],  # Spanish uses Latin
        'fr': [(0x0041, 0x007A)],  # French uses Latin
    }
    
    # Common words/phrases for quick detection
    LANGUAGE_KEYWORDS = {
        'hi': [
            'नमस्ते', 'धन्यवाद', 'कैसे', 'हैं', 'आप', 'मैं', 'हूँ', 
            'क्या', 'जी', 'हाँ', 'नहीं', 'ठीक', 'अच्छा'
        ],
        'en': [
            'hello', 'thanks', 'how', 'are', 'you', 'i', 'am',
            'what', 'yes', 'no', 'okay', 'good', 'please'
        ],
        'es': ['hola', 'gracias', 'cómo', 'estás'],
        'fr': ['bonjour', 'merci', 'comment', 'allez-vous']
    }
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
    
    def detect_input_language(self, text: str, fallback: str = 'en') -> str:
        """
        Detect language using multi-stage approach
        
        Args:
            text: Input text to analyze
            fallback: Default language if detection fails
            
        Returns:
            Language code (en, hi, es, fr, etc.)
        """
        if not text or len(text.strip()) < 2:
            return fallback
        
        # Clean text
        clean_text = self._preprocess_text(text)
        
        # Multi-stage detection
        detection_methods = [
            self._detect_by_script,
            self._detect_by_keywords,
            self._detect_by_langdetect,
        ]
        
        for method in detection_methods:
            try:
                result = method(clean_text)
                if result and result != 'unknown':
                    return result
            except:
                continue
        
        return fallback
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove URLs, emails, special characters (keep language chars)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)  # Keep Devanagari
        return text.strip()
    
    def _detect_by_script(self, text: str) -> Optional[str]:
        """Detect language by character script/unicode range"""
        sample = text[:100]  # Check first 100 chars
        
        for lang_code, ranges in self.SCRIPT_RANGES.items():
            for start, end in ranges:
                for char in sample:
                    if start <= ord(char) <= end:
                        return lang_code
        return None
    
    def _detect_by_keywords(self, text: str) -> Optional[str]:
        """Detect language by common keywords"""
        text_lower = text.lower()
        
        for lang_code, keywords in self.LANGUAGE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return lang_code
        return None
    
    def _detect_by_langdetect(self, text: str) -> Optional[str]:
        """Use langdetect library for statistical detection"""
        try:
            # Get probabilities for all languages
            from langdetect import detect_langs
            
            try:
                languages = detect_langs(text)
                if languages:
                    # Return language with highest probability
                    best_lang = max(languages, key=lambda x: x.prob)
                    if best_lang.prob >= self.confidence_threshold:
                        return best_lang.lang
            except:
                # Fallback to simple detect
                return langdetect.detect(text)
        except (lang_detect_exception.LangDetectException, Exception):
            pass
        
        return None
    
    def get_detection_confidence(self, text: str, language: str) -> float:
        """Calculate confidence score for detection"""
        if not text:
            return 0.0
        
        # Simple confidence calculation
        text_lower = text.lower()
        keywords = self.LANGUAGE_KEYWORDS.get(language, [])
        
        if keywords:
            matches = sum(1 for kw in keywords if kw in text_lower)
            confidence = min(1.0, matches / len(keywords) * 2)
            return round(confidence, 2)
        
        return 0.5  # Default medium confidence

# Global instance for easy import
detector = LanguageDetector()

# ==========================================
# MAIN FUNCTIONS FOR IMPORT
# ==========================================

def detect_input_language(text: str) -> str:
    """Simple wrapper for backward compatibility"""
    return detector.detect_input_language(text)


def get_system_prompt(language: str = "en", username: str = None) -> str:
    """
    Get system prompt based on language and username
    
    Args:
        language: Language code (en, hi, etc.)
        username: Optional username for personalization
        
    Returns:
        System prompt string
    """
    # Default prompts
    prompts = {
        "en": f"""You are AumCore AI{' (' + username + ')' if username else ''}, an advanced AI assistant specializing in programming, 
        system design, and technical solutions. Provide detailed, accurate, and 
        professional responses.""",
        
        "hi": f"""आप AumCore AI{' (' + username + ')' if username else ''} हैं, एक उन्नत AI सहायक जो प्रोग्रामिंग, सिस्टम डिज़ाइन और 
        तकनीकी समाधानों में विशेषज्ञ है। विस्तृत, सटीक और पेशेवर प्रतिक्रियाएँ दें।""",
        
        "es": f"""Eres AumCore AI{' (' + username + ')' if username else ''}, un asistente de IA avanzado especializado en programación,
        diseño de sistemas y soluciones técnicas. Proporciona respuestas detalladas,
        precisas y profesionales.""",
        
        "fr": f"""Vous êtes AumCore AI{' (' + username + ')' if username else ''}, un assistant IA avancé especializado en programación,
        conception de systèmes et solutions técnicas. Fournissez des réponses détaillées,
        précises et professionnelles."""
    }
    
    return prompts.get(language, prompts["en"])


def detect_with_confidence(text: str) -> Tuple[str, float]:
    """Detect language with confidence score"""
    detector_obj = LanguageDetector()
    language = detector_obj.detect_input_language(text)
    confidence = detector_obj.get_detection_confidence(text, language)
    return language, confidence


def generate_basic_code(task):
    """Generate basic code templates - TEMPORARY SIMPLE VERSION"""
    task_lower = task.lower()
    
    if 'drive' in task_lower or 'mount' in task_lower:
        return "```python\nfrom google.colab import drive\ndrive.mount('/content/gdrive')\n```"
    elif 'web' in task_lower or 'app' in task_lower:
        return "```python\nfrom fastapi import FastAPI\napp = FastAPI()\n@app.get('/')\ndef home(): return {'message': 'Hello'}\n```"
    else:
        return "```python\nprint('Hello from AumCore AI')\n```"


# Module metadata
__version__ = "2.0.1"
__author__ = "AumCore AI"
__all__ = [
    'detect_input_language',
    'get_system_prompt',
    'detect_with_confidence', 
    'LanguageDetector',
    'generate_basic_code'
]