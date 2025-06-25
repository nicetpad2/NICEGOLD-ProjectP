
# Minimal FastAPI app for serving
# serve_app.py
from fastapi import FastAPI
import os
app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok", "message": "FastAPI server is running!"}