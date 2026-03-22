Audio Embeddings API (DevOps Project)

A containerized microservice that generates and searches audio embeddings using the YAMNet deep learning model.

Tech Stack

Language: Python 3.12

Framework: FastAPI

ML Engine: TensorFlow 2.18 / TensorFlow Hub

Database: SQLite (auto-initialized schema)

Containerization: Docker & Docker Compose (AMD64)

Engineering Highlights

Environment Isolation: Resolved dependency conflicts between Python 3.12 and TensorFlow Hub using custom Docker build steps

Reliability: Implemented Docker healthchecks for service monitoring

Stability: Added automated API tests executed inside the container

Quick Start
docker compose up --build -d


Note: First startup may take ~1 minute due to YAMNet model loading.

API Endpoints
POST /embeddings

Upload a .wav file to generate and store a 1024-dimensional embedding.

POST /search

Search for similar audio files using cosine similarity.

GET /docs

Interactive Swagger UI.

Testing

Run tests inside the running container:

docker exec -it audio_embeddings_api pytest test_main.py
