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

```bash
docker exec -it audio_embeddings_api pytest test_main.py
```

## Data Verification

To verify that embeddings are successfully stored in the database, run:

```bash
docker compose exec api python -c "import sqlite3; conn = sqlite3.connect('/app/embeddings.db'); print(conn.cursor().execute('SELECT filename FROM audio_data').fetchall()); conn.close()"
```

Arendaja märkused (Developer Notes)

Projekti väljakutsed ja lahendused:
Selle projekti seadistamine Python 3.12 keskkonnas osutus üsna keerukaks. Peamiseks takistuseks oli tensorflow-hub ühilduvus, kuna see sõltus vananenud pkg_resources moodulist, mida uuemates Pythoni versioonides enam vaikimisi ei ole.

Lahendus:
Kasutasin Dockeris eraldi build-etappi, et tagada setuptools==69.5.1 paigaldamine. See võimaldas kasutada kaasaegset Pythoni versiooni ning samal ajal käivitada YAMNet mudelit stabiilselt. Tegemist on hea näitega sellest, kuidas DevOps-lähenemine aitab ületada pärandtarkvaraga seotud piiranguid.

Märkus: Kui logides kuvatakse CUDA-ga seotud hoiatusi, siis ei ole põhjust muretsemiseks – süsteem lülitub automaatselt CPU kasutamisele, mis on Maci riistvara puhul ootuspärane käitumine.
