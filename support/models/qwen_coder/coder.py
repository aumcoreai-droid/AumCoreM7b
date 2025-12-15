from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QwenCoder:
    # Qwen2 मॉडल का नाम इस्तेमाल करें
    # (Using Qwen2 model name)
    MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"

    def __init__(self, device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        self.model = None
        logger.info(f"QwenCoder initialized. Device set to: {self.device}")

    def load_model(self):
        """Model और Tokenizer को लोड करता है।"""
        try:
            # 1. Tokenizer लोड करें
            self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            logger.info("Tokenizer loaded successfully.")

            # 2. Model लोड करें (Device के अनुसार)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME,
                torch_dtype=torch.bfloat16 if self.device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16,
                device_map=self.device
            )
            self.model.eval()
            logger.info(f"Model {self.MODEL_NAME} loaded on {self.device}.")

        except Exception as e:
            logger.error(f"QwenCoder Model loading failed: {e}")
            self.model = None

    def generate_code(self, prompt: str) -> str:
        """दिए गए प्रॉम्प्ट के आधार पर कोड जनरेट करता है।"""
        if not self.model:
            return "Error: Qwen Model not loaded."
        
        # प्रॉम्प्ट को सही फॉर्मेट में रखें (Simple instruction format)
        messages = [
            {"role": "system", "content": "You are a professional coding assistant that writes only perfect Python code."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # कोड जनरेट करें
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        generated_ids = [
            output_id[len(model_inputs["input_ids"][0]):] for output_id in generated_ids
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

# यदि आप इसे अकेले चलाना चाहें तो
if __name__ == "__main__":
    coder = QwenCoder()
    coder.load_model()
    # test_prompt = "Write a Python function to calculate the factorial of a number."
    # code = coder.generate_code(test_prompt)
    # print(f"Generated Code: 
{code}")