
# config.py - Without tokens
import os

class Settings:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    TIDB_HOST = "gateway01.ap-southeast-1.prod.aws.tidbcloud.com"
    TIDB_PORT = 4000
    TIDB_USER = "2Rg6kfo2rNEB3PN.root"
    TIDB_PASSWORD = os.getenv("TIDB_PASSWORD")
    TIDB_DATABASE = "test"

settings = Settings()
