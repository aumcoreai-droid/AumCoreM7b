import logging
import os
import json
from datetime import datetime

# अपने कोर कॉम्पोनेंट्स को इंपोर्ट करें (Import your core components)
from support.models.qwen_coder.coder import QwenCoder
from core.memory.chroma_adapter import ChromaAdapter
from support.tools.file_manager import FileManager
from support.tools.image_reader import ImageReader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AUMCORE")

class AICore:
    def __init__(self):
        # 1. मॉडल और मेमोरी को इनिशियलाइज़ करें (Initialize Model and Memory)
        logger.info("Initializing AICore components...")
        self.coder = QwenCoder()
        self.memory = ChromaAdapter()
        
        # 2. टूल्स को इनिशियलाइज़ करें (Initialize Tools)
        self.file_manager = FileManager(base_dir='ai_workspace')
        self.image_reader = ImageReader()
        
        # 3. टूल्स को एक डिक्शनरी में समूहित करें ताकि AI उन्हें एक्सेस कर सके (Group tools for AI access)
        self.tools = {
            "file_manager": self.file_manager,
            "image_reader": self.image_reader,
            "memory": self.memory
        }

    def start_system(self):
        """सभी ज़रूरी कॉम्पोनेंट्स को लोड और शुरू करता है।"""
        logger.info("Starting AumCore System...")
        
        # 1. मॉडल लोड करें
        self.coder.load_model()
        
        # 2. मेमोरी DB इनिशियलाइज़ करें
        self.memory.initialize_db()
        
        # 3. सिस्टम और यूजर लक्ष्यों को मेमोरी में जोड़ें (SDS System)
        self.seed_memory()
        
        logger.info("AumCore System fully operational.")

    def seed_memory(self):
        """ज़रूरी प्रारंभिक डेटा को SDS (ChromaDB) में जोड़ें।"""
        initial_docs = [
            "User's primary goal: Build a real, self-coding AI, not a simple chatbot.",
            "Key features to implement: Image Read (OCR/Visual), .txt/.doc File Crafting, Hindi/English Reply, SDS System (Self-Data Storage).",
            "Current branch for development: aicore-refactor-phase2.",
            f"System Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        # इन डॉक्युमेंट्स को मेमोरी में जोड़ें
        if self.memory.client:
            self.memory.add_data(
                documents=initial_docs,
                metadatas=[{"source": "system_goal", "priority": "high"}] * len(initial_docs),
                ids=[f"sds_seed_{i+1}" for i in range(len(initial_docs))]
            )
            logger.info(f"Added {len(initial_docs)} seed items to SDS Memory.")
            
    def run_interaction(self, user_prompt: str) -> str:
        """यूजर प्रॉम्प्ट को प्रोसेस करता है, मेमोरी से संदर्भ लेता है, और Qwen Coder का उपयोग करता है।"""
        logger.info(f"Processing user prompt: {user_prompt[:50]}...")
        
        # 1. मेमोरी से प्रासंगिक संदर्भ क्वेरी करें
        context_results = self.memory.query_memory(user_prompt, n_results=3)
        context = "Relevant past context/goals: " + " | ".join(context_results.get('documents', [[]])[0])

        # 2. Qwen Coder के लिए अंतिम प्रॉम्प्ट बनाएं
        full_prompt = (
            f"CONTEXT: {context}\n"
            f"AVAILABLE TOOLS: {list(self.tools.keys())}\n"
            f"TASK (Write Python code/text based on the user's request): {user_prompt}"
        )
        
        # 3. Qwen Coder से प्रतिक्रिया जनरेट करें
        ai_response = self.coder.generate_code(full_prompt)
        
        # 4. प्रतिक्रिया को प्रोसेस करें (उदाहरण के लिए, अगर यह कोई टूल कमांड है तो उसे चलाएं, लेकिन अभी सिर्फ़ आउटपुट दें)
        
        return ai_response

# --- सिस्टम का स्टार्टअप (System Startup) ---
if __name__ == "__main__":
    logger.info("Starting AumCore AI application...")
    
    # AICore इंस्टेंस बनाएं
    core = AICore()
    
    # सभी कॉम्पोनेंट्स लोड करें
    core.start_system()
    
    # एक शुरुआती इंटरैक्शन चलाएं (Demo Interaction)
    # response = core.run_interaction("Define the current state and goals of AumCore AI, focusing on the self-coding aspect.")
    # print("\n--- AI RESPONSE ---\n")
    # print(response)
    
    # कंटेनर को चलते रहने दें (Keep the container running)
    logger.info("AumCore AI is running and ready for deployment interaction.")