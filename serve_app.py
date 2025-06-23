# serve_app.py
# Minimal FastAPI app for serving
import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "FastAPI server is running!"}
