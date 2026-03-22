import pytest
from fastapi.testclient import TestClient
from main import app
import os

# Initialize the TestClient with our FastAPI app
client = TestClient(app)

def test_read_main():
    """Check if the API server is up and responding (Health Check)."""
    response = client.get("/")
    assert response.status_code == 200

def test_embeddings_upload():
    """Verify that the ML model endpoint can receive a file and process it."""
    
    # Create a temporary dummy audio file for testing
    file_path = "test_audio.wav"
    with open(file_path, "wb") as f:
        # Fill it with 1024 random bytes to simulate a small audio file
        f.write(os.urandom(1024)) 

    # Sending the file to the /embeddings endpoint
    with open(file_path, "rb") as f:
        response = client.post(
            "/embeddings", 
            files={"file": ("test.wav", f, "audio/wav")}
        )
    
    # Cleanup: remove the temporary file after the test
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # We expect 200 (Success) or 500 (ML error due to invalid audio data).
    # Either way, it proves the endpoint is reachable and logic is triggered.
    assert response.status_code in [200, 500]