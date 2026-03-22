# Audio Embeddings API (DevOps Project)

A containerized microservice that generates and searches audio embeddings using the **YAMNet** deep learning model.

## Tech Stack

- **Language:** Python 3.12
- **Framework:** FastAPI
- **ML Engine:** TensorFlow 2.18 / TensorFlow Hub
- **Database:** SQLite (with automatic schema initialization)
- **Containerization:** Docker & Docker Compose (AMD64 platform)

## Engineering Highlights

- **Environment Isolation:** Successfully resolved dependency conflicts between Python 3.12 and legacy `tensorflow-hub` using custom build steps in Docker.
- **Reliability:** Integrated Docker **Healthchecks** to monitor service availability.
- **Stability:** Automated API testing implemented within the container environment.

## Quick Start

1. **Build and Start:**
   ```bash
   docker compose up --build -d
   Note: The first start may take ~1 minute to load the heavy YAMNet model.
   ```

Access API Documentation:
Open http://localhost:8000/docs in your browser.

Features
POST /embeddings: Upload a .wav file to generate and store a 1024-dimensional vector in the local database.

POST /search: Find the most similar audio file in the database using Cosine Similarity.

GET /docs: Fully interactive Swagger documentation.

Testing
To run infrastructure and logic tests inside the running Docker container, use the following command:

Bash
docker exec -it audio_embeddings_api pytest test_main.py
