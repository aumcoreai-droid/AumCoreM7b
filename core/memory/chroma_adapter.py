import chromadb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaAdapter:
    # लोकल डिस्क पर डेटाबेस स्टोर करने का पाथ
    # (Path to store the database on local disk)
    DB_PATH = "support/data/chroma_db"
    
    # AI की लंबी अवधि की मेमोरी के लिए कलेक्शन का नाम
    # (Collection name for AI's long-term memory)
    COLLECTION_NAME = "aumcore_long_term_memory"

    def __init__(self):
        self.client = None
        self.collection = None
        logger.info(f"ChromaAdapter initialized. DB Path: {self.DB_PATH}")

    def initialize_db(self):
        """ChromaDB क्लाइंट को इनिशियलाइज़ करता है और कलेक्शन बनाता है।"""
        try:
            # PersistentClient का उपयोग करें ताकि डेटा कंटेनर बंद होने के बाद भी बचा रहे
            self.client = chromadb.PersistentClient(path=self.DB_PATH)
            
            # कलेक्शन बनाएं या पाएं (Get or create collection)
            # Embedding function default पर सेट है, जिसे बाद में ऑप्टिमाइज़ किया जा सकता है।
            self.collection = self.client.get_or_create_collection(self.COLLECTION_NAME)
            
            logger.info(f"ChromaDB client connected. Collection '{self.COLLECTION_NAME}' ready.")
            return True
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            self.client = None
            return False

    def add_data(self, documents: list[str], metadatas: list[dict], ids: list[str]):
        """वेक्टर डेटाबेस में नए दस्तावेज़ जोड़ता है।"""
        if not self.collection:
            logger.error("ChromaDB not initialized. Cannot add data.")
            return

        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added {len(documents)} documents to ChromaDB.")
        except Exception as e:
            logger.error(f"Error adding data to ChromaDB: {e}")

    def query_memory(self, query_text: str, n_results: int = 5) -> list:
        """मेमोरी से सबसे प्रासंगिक जानकारी (relevant information) क्वेरी करता है।"""
        if not self.collection:
            logger.error("ChromaDB not initialized. Cannot query memory.")
            return []

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            logger.info(f"Query successful. Found {len(results.get('documents', [[]])[0])} results.")
            return results
        except Exception as e:
            logger.error(f"Error querying ChromaDB: {e}")
            return []

# यदि आप इसे अकेले चलाना चाहें तो
if __name__ == "__main__":
    adapter = ChromaAdapter()
    if adapter.initialize_db():
        adapter.add_data(
            documents=["The user prefers Hindi and English replies.", "The main goal is to build a real AI, not a chatbot."],
            metadatas=[{"source": "user_memory"}, {"source": "user_goal"}],
            ids=["mem_001", "mem_002"]
        )
        # relevant_info = adapter.query_memory("What is the user's coding goal?")
        # print("Query Results:", relevant_info)