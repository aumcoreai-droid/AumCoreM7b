
from fastapi import FastAPI
from core.debug_router import handle_debug_request

app = FastAPI(title="AumCoreM7b SDS", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "AumCoreM7b Smart Debugging System"}

@app.post("/debug")
async def debug_endpoint(payload: dict):
    try:
        result = handle_debug_request(payload)
        return {"success": True, "data": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
