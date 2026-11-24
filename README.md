
# 🎬 Netflix Personalized Top Picks

A personalized movie and TV show recommendation system powered by a **Two-Tower Deep Learning Model**, **Qdrant Vector Database**, and **FastAPI**.

The project aims to recommend the top 10 titles to each member based on their viewing history and learned preferences.

## 🚀 Project Overview

Netflix Personalized Top Picks generates personalized recommendations instead of relying only on globally popular titles.

### Key Features

- Personalized movie and TV show recommendations
- Two-Tower deep learning architecture
- Member and title embeddings
- 64-dimensional vector representations
- Qdrant similarity search
- FastAPI recommendation endpoint
- Previously watched title exclusion
- Top 10 recommendations per member
- API response validation and automated testing

## 🏗️ System Architecture

```text
Viewing History + Title Catalog
              |
              v
       Data Preparation
              |
              v
       Two-Tower Model
         /         \
        v           v
   User Tower    Item Tower
        |           |
        v           v
 Member Vector   Title Vectors
        |           |
        |        Qdrant DB
        |           |
        +-----+-----+
              |
              v
       Similarity Search
              |
              v
     Watched Title Exclusion
              |
              v
         FastAPI
              |
              v
     Top 10 Recommendations
```

## 📌 Project Stages

### Stage 1: Data Preparation

- Clean viewing history.
- Validate member and title IDs.
- Prepare catalog features.
- Review data quality.
- Separate training and evaluation data.

### Stage 2: Two-Tower Model

- Build User Tower and Item Tower.
- Train member and title embeddings.
- Use 64-dimensional embeddings.
- Learn personalized member preferences.
- Evaluate against a popularity baseline.

### Stage 3: Qdrant + Recommendation API

- Reuse the trained Stage 2 model.
- Generate title embeddings.
- Store title vectors in Qdrant.
- Generate member vectors.
- Retrieve similar titles.
- Exclude previously watched titles.
- Return exactly 10 recommendations through FastAPI.

**Note:** Stage 3 implementation and verification status must be updated after execution.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning |
| Pandas | Data processing |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning |
| Qdrant | Vector database |
| FastAPI | API development |
| Pydantic | Data validation |
| Pytest | Testing |
| JSON Schema | API contract validation |
| Git & GitHub | Version control |

## 📂 Project Structure

```text
netflix-personalized-top-picks/
│
├── data/
├── stage2_output/
│   ├── model_final.pt
│   └── locked_epoch.json
│
├── samples/
│   ├── top_picks_200_M-<id>.json
│   └── top_picks_404_M-99999.json
│
├── scripts/
│   └── load_title_vectors_to_qdrant.py
│
├── app/
│   └── main.py
│
├── train.py
├── recsys_data.py
├── evaluate.py
├── test_pipeline.py
├── requirements.txt
├── .env.example
├── data_quality_report.md
├── stage3_artifact_inventory.md
├── stage3_implementation_note.md
└── README.md
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Asifpatan05/netflix-personalized-top-picks.git
cd netflix-personalized-top-picks
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🔐 Environment Configuration

Create a `.env` file using `.env.example`.

```env
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=netflix_title_embeddings

MODEL_DIR=stage2_output
CATALOG_PATH=data/netflix_title_catalog_2026-11.csv
EVENTS_PATH=cleaned_viewing_history.csv
```

Never upload API keys or `.env` files to GitHub.

## 🗄️ Start Qdrant

Run Qdrant locally using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Expected collection:

```text
netflix_title_embeddings
```

Distance metric: Cosine

Embedding dimension: 64

Expected catalog size: 300 titles. Verify the actual loaded count after running the loader.

## 🚀 Run the Recommendation API

After the Stage 3 implementation and required artifacts are available:

```bash
uvicorn app.main:app --reload
```

API endpoint:

```http
GET /recommendations/{user_id}
```

Example:

```text
http://127.0.0.1:8000/recommendations/M-10001
```

## 📡 API Response Examples

### HTTP 200 — Successful Recommendation

```json
{
  "user_id": "M-10001",
  "recommendations": [
    {
      "title_id": "TT-0001",
      "title_name": "Example Title",
      "primary_genre": "Drama",
      "score": 0.95
    }
  ],
  "latency_ms": 12.5
}
```

This is a response format illustration, not a real API result. The actual response must contain exactly 10 recommendations.

### HTTP 404 — Member Not Found

Request:

```text
GET /recommendations/M-99999
```

Response:

```json
{
  "detail": "Member not found",
  "user_id": "M-99999"
}
```

## 🧪 Testing

Run the test suite:

```bash
pytest -v
```

Tests should verify:

- Model artifact loading
- Qdrant collection creation
- Title-vector dimensions
- Title ID mapping
- Idempotent vector loading
- Previously watched title exclusion
- Exactly 10 unique recommendations
- Descending recommendation scores
- Valid catalog metadata
- HTTP 200 and HTTP 404 responses
- JSON Schema validation

## 📄 API Sample Files

Save actual API responses inside `samples/`.

```text
samples/
├── top_picks_200_M-<id>.json
└── top_picks_404_M-99999.json
```

The successful response must come from a real member. The 404 response must come from an actual unknown-member API request.

## 🔒 Recommendation Rules

1. Reuse the existing Stage 2 checkpoint.
2. Do not retrain the model during Stage 3.
3. Do not use holdout events for member histories.
4. Exclude titles played before the training cutoff.
5. Recommend only titles present in the catalog.
6. Keep zero-play titles eligible.
7. Return exactly 10 unique recommendations.
8. Follow the defined API response contract.

Training cutoff: `2026-11-10`

## 📊 Evaluation

Compare personalized recommendations with the popularity baseline using:

- The same held-out week
- The same eligible members
- The same top 10 rules

This makes the comparison fair and measures whether personalized recommendations improve recommendation quality.

## 🌱 Future Enhancements

- Member-facing Top Picks interface
- Recommendation explanations
- Cold-start recommendation improvements
- API performance monitoring
- Recommendation quality monitoring
- Cloud deployment

## 👨‍💻 Author

**Asif Patan**

GitHub: [Asifpatan05](https://github.com/Asifpatan05)

Project Repository:

[Netflix Personalized Top Picks](https://github.com/Asifpatan05/netflix-personalized-top-picks)

---

**Project:** Netflix Personalized Top Picks  
**Model:** Two-Tower Recommendation Model  
**Vector Database:** Qdrant  
**API:** FastAPI  
**Language:** Python
