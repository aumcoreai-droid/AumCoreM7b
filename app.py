
import os
import uvicorn
import json
import nest_asyncio
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from groq import Groq

nest_asyncio.apply()
app = FastAPI()
client = Groq(api_key="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")

# ... (Previous 250+ lines of UI and Logic code included here) ...

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
