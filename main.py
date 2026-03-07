from fastapi import FastAPI, UploadFile, File
from typing import List
import os
import shutil
import numpy as np
from model import yamnet
from database import save_embedding, get_all_embeddings

app = FastAPI(title="Audio Embeddings API", description="AI-powered audio similarity search")

@app.get("/")
async def root():
    return {"message": "Audio Embeddings API is running. Visit /docs for documentation."}

@app.post("/embeddings")
async def create_embeddings(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        embedding = yamnet.process_audio(temp_path)
        save_embedding(file.filename, embedding)
        return {"status": "success", "filename": file.filename, "message": "Processed and saved"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/search")
async def search_embeddings(file: UploadFile = File(...), top_k: int = 5):
    temp_path = f"search_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        query_vector = np.array(yamnet.process_audio(temp_path))
        all_data = get_all_embeddings()
        
        if not all_data:
            return {"message": "Database is empty. Please upload some files first."}

        scores = []
        for item in all_data:
            stored_vector = np.array(item["embedding"])
            similarity = np.dot(query_vector, stored_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(stored_vector)
            )
            scores.append({
                "filename": item["filename"],
                "similarity": round(float(similarity), 4)
            })

        scores.sort(key=lambda x: x["similarity"], reverse=True)
        return scores[:top_k]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)