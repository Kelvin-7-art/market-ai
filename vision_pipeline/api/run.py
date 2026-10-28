# api/run.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,      # set True during dev
        workers=1,         # multiple workers = multiple model copies
    )