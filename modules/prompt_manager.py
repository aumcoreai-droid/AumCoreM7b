"""
Prompt Management System for AumCore AI
Version: 4.0.0
Author: AumCore AI
Location: /app/modules/prompt_manager.py
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import hashlib

class ConversationStyle(Enum):
    """Different conversation styles"""
    CONCISE = "concise"      # Short, direct answers
    DETAILED = "detailed"    # Thorough explanations
    TECHNICAL = "technical"  # Code-focused, precise
    FRIENDLY = "friendly"    # Casual, conversational
    PROFESSIONAL = "professional"  # Formal, business-like

class LanguageCode(Enum):
    """Supported language codes"""
    ENGLISH = "en"
    HINDI = "hi"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"

@dataclass
class PromptConfig:
    """Configuration for generating prompts"""
    language: LanguageCode = LanguageCode.ENGLISH
    style: ConversationStyle = ConversationStyle.CONCISE
    username: Optional[str] = None
    context_length: int = 5  # Number of previous messages to include
    include_code_examples: bool = False
    temperature: float = 0.7
    max_response_length: int = 1000

class AumCorePromptManager:
    """
    Advanced Prompt Management System
    Handles multi-language prompts, conversation context, and response optimization
    """
    
    def __init__(self, prompts_dir: str = "data/prompts"):
        self.prompts_dir = prompts_dir
        self._prompt_cache: Dict[str, str] = {}
        self._conversation_history: List[Dict] = []
        self._load_base_prompts()
        
        # Create prompts directory if it doesn't exist
        os.makedirs(self.prompts_dir, exist_ok=True)
    
    def _load_base_prompts(self):
        """Load base system prompts for all languages and styles"""
        self._base_prompts = {
            "system": {
                "en": {
                    "concise": """You are {name}, an AI assistant. Answer directly and clearly.""",
                    "detailed": """You are {name}, an advanced AI assistant. Provide comprehensive, well-explained responses with examples when helpful.""",
                    "technical": """You are {name}, an AI specializing in programming and technology. Provide precise, code-focused answers with best practices.""",
                    "friendly": """You are {name}, a friendly AI assistant. Be conversational, helpful, and approachable in your responses.""",
                    "professional": """You are {name}, a professional AI assistant. Maintain formal tone, accuracy, and clarity in all responses."""
                },
                "hi": {
                    "concise": """आप {name} हैं, एक AI सहायक। सीधे और स्पष्ट उत्तर दें।""",
                    "detailed": """आप {name} हैं, एक उन्नत AI सहायक। विस्तृत, समझदार उत्तर दें और आवश्यक होने पर उदाहरण दें।""",
                    "technical": """आप {name} हैं, प्रोग्रामिंग और तकनीक में विशेषज्ञ AI। सटीक, कोड-केंद्रित उत्तर दें और बेस्ट प्रैक्टिस बताएं।""",
                    "friendly": """आप {name} हैं, एक मित्रवत AI सहायक। बातचीत के अंदाज में, सहायक और आसानी से संपर्क करने योग्य रहें।""",
                    "professional": """आप {name} हैं, एक पेशेवर AI सहायक। औपचारिक शैली, सटीकता और स्पष्टता बनाए रखें।"""
                }
            },
            "greeting": {
                "en": {
                    "morning": "Good morning! I'm {name}. How can I assist you today?",
                    "afternoon": "Good afternoon! I'm {name}. What can I help you with?",
                    "evening": "Good evening! I'm {name}. How may I be of service?",
                    "general": "Hello! I'm {name}. How can I help you?"
                },
                "hi": {
                    "morning": "सुप्रभात! मैं {name} हूँ। आज आपकी कैसे सहायता कर सकता हूँ?",
                    "afternoon": "नमस्ते! मैं {name} हूँ। आपकी क्या सहायता कर सकता हूँ?",
                    "evening": "शुभ संध्या! मैं {name} हूँ। मैं आपकी कैसे सेवा कर सकता हूँ?",
                    "general": "नमस्ते! मैं {name} हूँ। मैं आपकी कैसे सहायता कर सकता हूँ?"
                }
            },
            "error_responses": {
                "en": {
                    "no_code_request": "I don't see a specific code request. Could you clarify what you need?",
                    "confused_context": "I want to make sure I understand correctly. Could you rephrase your question?",
                    "technical_help": "I'd be happy to help with technical questions. What specifically do you need?"
                },
                "hi": {
                    "no_code_request": "मुझे कोई विशिष्ट कोड अनुरोध नहीं दिख रहा। क्या आप स्पष्ट कर सकते हैं कि आपको क्या चाहिए?",
                    "confused_context": "मैं सुनिश्चित करना चाहता हूँ कि मैं सही समझ रहा हूँ। क्या आप अपना प्रश्न दोबारा कह सकते हैं?",
                    "technical_help": "मैं तकनीकी प्रश्नों में मदद करने में खुशी होगी। आपको विशेष रूप से क्या चाहिए?"
                }
            }
        }
    
    def _get_cache_key(self, config: PromptConfig, category: str) -> str:
        """Generate cache key for prompt"""
        key_data = f"{category}:{config.language.value}:{config.style.value}:{config.username}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_system_prompt(self, 
                         language: str = "en", 
                         username: str = None,
                         style: str = "concise") -> str:
        """
        Get system prompt for AI interaction
        
        Args:
            language: Language code (en, hi, etc.)
            username: Optional username for personalization
            style: Conversation style (concise, detailed, technical, friendly, professional)
            
        Returns:
            System prompt string
        """
        # Normalize inputs
        lang = language.lower() if language else "en"
        style_key = style.lower() if style else "concise"
        
        # Get name part
        name_part = f"{username}'s AI" if username else "AumCore AI"
        
        # Get base prompt
        try:
            base_prompt = self._base_prompts["system"][lang][style_key]
        except KeyError:
            # Fallback to English concise
            base_prompt = self._base_prompts["system"]["en"]["concise"]
        
        # Format with name
        return base_prompt.format(name=name_part)
    
    def get_context_aware_prompt(self,
                                user_message: str,
                                language: str = "en",
                                username: str = None,
                                previous_messages: List[Dict] = None) -> str:
        """
        Get prompt with conversation context awareness
        
        Args:
            user_message: Current user message
            language: Language code
            username: Optional username
            previous_messages: List of previous conversation messages
            
        Returns:
            Context-aware prompt string
        """
        system_prompt = self.get_system_prompt(language, username, "friendly")
        
        # Build context if available
        context_part = ""
        if previous_messages and len(previous_messages) > 0:
            context_part = "\n\nPrevious conversation:\n"
            for msg in previous_messages[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                context_part += f"{role}: {content}\n"
        
        # Analyze message for special handling
        message_lower = user_message.lower()
        
        if "code" in message_lower or "program" in message_lower or "python" in message_lower:
            style = "technical"
        elif len(user_message.split()) < 4:  # Very short message
            style = "concise"
        else:
            style = "detailed"
        
        # Add style instruction
        style_instruction = ""
        if style == "technical":
            style_instruction = "Focus on providing accurate code and technical explanations."
        elif style == "concise":
            style_instruction = "Keep the response brief and to the point."
        
        final_prompt = f"""{system_prompt}
        {style_instruction}
        {context_part}
        
        Current user message: {user_message}
        
        Please respond appropriately based on the context and message."""
        
        return final_prompt
    
    def get_greeting(self, 
                    language: str = "en", 
                    username: str = None,
                    time_of_day: str = None) -> str:
        """
        Get appropriate greeting based on time of day
        
        Args:
            language: Language code
            username: Optional username
            time_of_day: Specific time of day (morning, afternoon, evening)
            
        Returns:
            Greeting message
        """
        lang = language.lower() if language else "en"
        
        # Determine time of day if not provided
        if not time_of_day:
            hour = datetime.now().hour
            if 5 <= hour < 12:
                time_of_day = "morning"
            elif 12 <= hour < 17:
                time_of_day = "afternoon"
            elif 17 <= hour < 22:
                time_of_day = "evening"
            else:
                time_of_day = "general"
        
        # Get greeting template
        try:
            greeting_templates = self._base_prompts["greeting"][lang]
            template = greeting_templates.get(time_of_day, greeting_templates["general"])
        except KeyError:
            # Fallback to English
            greeting_templates = self._base_prompts["greeting"]["en"]
            template = greeting_templates.get(time_of_day, greeting_templates["general"])
        
        name_part = f"{username}'s AI" if username else "AumCore AI"
        return template.format(name=name_part)
    
    def detect_response_style_needed(self, user_message: str) -> Dict:
        """
        Analyze user message to determine appropriate response style
        
        Args:
            user_message: User's input text
            
        Returns:
            Dictionary with style recommendations
        """
        message_lower = user_message.lower()
        words = user_message.split()
        
        analysis = {
            "language": "en",  # Default, will be detected elsewhere
            "style": "detailed",
            "needs_code": False,
            "is_technical": False,
            "is_casual": False,
            "word_count": len(words)
        }
        
        # Check for technical/code requests
        code_keywords = ["code", "program", "function", "script", "algorithm", 
                        "python", "javascript", "java", "html", "css", "sql",
                        "error", "bug", "debug", "compile", "syntax"]
        
        if any(keyword in message_lower for keyword in code_keywords):
            analysis["needs_code"] = True
            analysis["is_technical"] = True
            analysis["style"] = "technical"
        
        # Check for casual conversation
        casual_keywords = ["hi", "hello", "hey", "how are you", "what's up",
                          "thanks", "thank you", "please", "ok", "okay"]
        
        if any(keyword in message_lower for keyword in casual_keywords):
            analysis["is_casual"] = True
            analysis["style"] = "friendly"
        
        # Check for very short messages
        if len(words) <= 3:
            analysis["style"] = "concise"
        
        # Check for complex questions
        question_words = ["how", "what", "why", "when", "where", "which", "explain", "describe"]
        if any(user_message.strip().startswith(word) for word in question_words):
            analysis["style"] = "detailed"
        
        return analysis
    
    def save_custom_prompt(self, 
                          category: str, 
                          language: str, 
                          style: str, 
                          prompt_text: str):
        """
        Save custom prompt to file
        
        Args:
            category: Prompt category (system, greeting, error_responses)
            language: Language code
            style: Prompt style
            prompt_text: Custom prompt text
        """
        custom_file = os.path.join(self.prompts_dir, "custom_prompts.json")
        
        # Load existing or create new
        if os.path.exists(custom_file):
            with open(custom_file, 'r', encoding='utf-8') as f:
                custom_prompts = json.load(f)
        else:
            custom_prompts = {}
        
        # Update structure
        if category not in custom_prompts:
            custom_prompts[category] = {}
        if language not in custom_prompts[category]:
            custom_prompts[category][language] = {}
        
        custom_prompts[category][language][style] = prompt_text
        
        # Save back
        with open(custom_file, 'w', encoding='utf-8') as f:
            json.dump(custom_prompts, f, indent=2, ensure_ascii=False)
        
        # Clear cache
        self._prompt_cache.clear()
    
    def add_to_conversation_history(self, role: str, content: str):
        """
        Add message to conversation history
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        self._conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 20 messages
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]
    
    def get_conversation_summary(self) -> str:
        """
        Get summary of recent conversation
        
        Returns:
            Conversation summary string
        """
        if not self._conversation_history:
            return "No recent conversation."
        
        summary = f"Recent conversation ({len(self._conversation_history)} messages):\n"
        for msg in self._conversation_history[-5:]:  # Last 5 messages
            role = msg["role"]
            content_preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            summary += f"{role}: {content_preview}\n"
        
        return summary
    
    def clear_conversation_history(self):
        """Clear all conversation history"""
        self._conversation_history = []

# Global instance for easy import
prompt_manager = AumCorePromptManager()

# Backward compatibility functions
def get_system_prompt(language: str = "en", username: str = None) -> str:
    """
    Legacy compatibility function
    
    Args:
        language: Language code
        username: Optional username
        
    Returns:
        System prompt string
    """
    return prompt_manager.get_system_prompt(language, username)

# Module exports
__all__ = [
    'AumCorePromptManager',
    'PromptConfig',
    'ConversationStyle', 
    'LanguageCode',
    'prompt_manager',
    'get_system_prompt'
]
# ============================================
# MODULE REGISTRATION FOR APPPY
# ============================================

def register_module(app, client, username):
    """
    Required function for ModuleManager to load this module
    """
    print("✅ Prompt Manager module registered with FastAPI")
    
    # You can add route registration here if needed
    # Example: 
    # @app.get("/prompt-manager/status")
    # async def status():
    #     return {"module": "prompt_manager", "status": "active"}
    
    return {
        "module": "prompt_manager",
        "status": "registered",
        "version": __version__,
        "description": "Advanced prompt management system"
    }
# ============================================
# MODULE REGISTRATION FOR APPPY
# ============================================

def register_module(app, client, username):
    """
    Required function for ModuleManager to load this module
    """
    print("✅ Code Intelligence module registered with FastAPI")
    
    return {
        "module": "code_intelligence",
        "status": "registered",
        "version": __version__,
        "description": "Advanced code analysis and intelligence system"
    }
__version__ = "4.0.0"
__author__ = "AumCore AI"