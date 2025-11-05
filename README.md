# search_similarity

A FastAPI-based reranking and paragraph search API using BAAI’s `bge-reranker-large` model.

## Setup

### 1. Clone the repo

### 2. Create virtual environment and activate
  python3 -m venv venv
  source venv/bin/activate

### 3. Start the server
  python3 -m uvicorn search_endpoint:app --reload --host 0.0.0.0 --port 8000

### 4. Submit paragraphs
  curl -X POST "http://127.0.0.1:8000/submit_paragraphs" \
  -H "Content-Type: application/json" \
  -d @path_to/example_paragraph_submission.json

### 5. Submit question for searching - example question below (matches example paragraphs)
  curl -X POST "http://127.0.0.1:8000/search" \           
  -H "Content-Type: application/json" \
  -d '{        
    "session_id": [session_id_from_paragraph_submission],     
    "question": "What did Goring believe the person whom he refused to work with in 1940 and 1941 would gain with further support?",
    "top_k": 2
  }'
