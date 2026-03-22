import os
import uuid
import numpy as np
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from model import yamnet
from database import save_embedding, get_all_embeddings

# (DevOps style)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Audio Embeddings API", description="AI-powered audio similarity search")

@app.get("/")
async def root():
    return {"message": "Audio Embeddings API is running. Visit /docs for documentation."}

@app.post("/embeddings")
async def create_embeddings(file: UploadFile = File(...)):
   
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail=f"File {file.filename} is not an audio file")

    file_id = str(uuid.uuid4())
    temp_path = f"temp_{file_id}_{file.filename}"
    
    try:
        
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024): 
                buffer.write(chunk)
        
        logger.info(f"Processing audio for: {file.filename}")
        embedding = yamnet.process_audio(temp_path)
        
        save_embedding(file.filename, embedding)
        return {"status": "success", "filename": file.filename, "message": "Processed and saved"}
    
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")
    finally:
        
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/search")
async def search_embeddings(file: UploadFile = File(...), top_k: int = 5):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Please upload an audio file")

    file_id = str(uuid.uuid4())
    temp_path = f"search_{file_id}_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
        
        query_vector = np.array(yamnet.process_audio(temp_path))
        all_data = get_all_embeddings()
        
        if not all_data:
            return {"message": "Database is empty."}

        # --- (NUMPY MAGIC) vector ---
        filenames = [item["filename"] for item in all_data]
        matrix = np.array([item["embedding"] for item in all_data])

        dot_product = np.dot(matrix, query_vector)
        norm_matrix = np.linalg.norm(matrix, axis=1)
        norm_query = np.linalg.norm(query_vector)
        
        similarities = dot_product / (norm_matrix * norm_query + 1e-9)

        results = [
            {"filename": filenames[i], "similarity": round(float(similarities[i]), 4)}
            for i in range(len(filenames))
        ]

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Search failed")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)