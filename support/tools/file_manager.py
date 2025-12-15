import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileManager:
    def __init__(self, base_dir: str = 'workspace'):
        # AI द्वारा बनाए गए कोड/फाइलों के लिए एक सुरक्षित कार्यक्षेत्र (safe workspace)
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"FileManager initialized. Base directory: {self.base_dir}")

    def get_full_path(self, relative_path: str) -> str:
        """बेस डायरेक्टरी के सापेक्ष पूर्ण पाथ प्राप्त करता है।"""
        # सुनिश्चित करें कि AI '..' का उपयोग करके बेस डायरेक्टरी से बाहर न जाए (security check)
        full_path = os.path.join(self.base_dir, relative_path)
        if not full_path.startswith(self.base_dir):
            raise ValueError("Access outside the workspace is denied.")
        return full_path

    def create_or_write_file(self, relative_path: str, content: str) -> bool:
        """एक फ़ाइल बनाता है या उसकी सामग्री को ओवरराइट करता है (.txt, .py, आदि)।"""
        try:
            full_path = self.get_full_path(relative_path)
            # सुनिश्चित करें कि डायरेक्टरी मौजूद है
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"File successfully written: {relative_path}")
            return True
        except ValueError as ve:
            logger.error(f"Security error: {ve}")
            return False
        except Exception as e:
            logger.error(f"Error writing file {relative_path}: {e}")
            return False

    def read_file(self, relative_path: str) -> str | None:
        """एक फ़ाइल की सामग्री पढ़ता है।"""
        try:
            full_path = self.get_full_path(relative_path)
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"File successfully read: {relative_path}")
            return content
        except FileNotFoundError:
            logger.warning(f"File not found: {relative_path}")
            return None
        except ValueError as ve:
            logger.error(f"Security error: {ve}")
            return None
        except Exception as e:
            logger.error(f"Error reading file {relative_path}: {e}")
            return None

    def list_files(self, relative_path: str = '') -> list[str]:
        """दिए गए पाथ में फ़ाइलों और फ़ोल्डरों की सूची बनाता है।"""
        try:
            full_path = self.get_full_path(relative_path)
            if os.path.isdir(full_path):
                return os.listdir(full_path)
            else:
                logger.warning(f"Path is not a directory: {relative_path}")
                return []
        except ValueError as ve:
            logger.error(f"Security error: {ve}")
            return []
        except Exception as e:
            logger.error(f"Error listing files in {relative_path}: {e}")
            return []

# यदि आप इसे अकेले चलाना चाहें तो
if __name__ == "__main__":
    fm = FileManager(base_dir='temp_ai_workspace')
    fm.create_or_write_file('test_code/hello.py', 'print("Hello, AI!")')
    content = fm.read_file('test_code/hello.py')
    print(f"Content read: {content}")
    print(f"Files in test_code: {fm.list_files('test_code')}")