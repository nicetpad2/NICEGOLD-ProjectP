
# run_fastapi_standalone.py
# รัน FastAPI app ด้วย import string เพื่อรองรับ reload/workers
    import uvicorn
if __name__ == "__main__":
    uvicorn.run("serve_app:app", host = "0.0.0.0", port = 8001, reload = True)