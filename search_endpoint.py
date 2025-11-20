from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import torch
from FlagEmbedding import FlagAutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification



app = FastAPI(title="BG ME 3 Search API", version="1.0.0")
# model = FlagAutoModel.from_finetuned('BAAI/bge-base-en-v1.5',
#                                       query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
#                                       use_fp16=True)

model_name = "BAAI/bge-reranker-large"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu" # default, can change as necessary
model.to(device)

# paragraph_store: Dict[str, List[dict]] = {}
paragraph_store: List[List[dict]] = []


def rerank_one(query: str, docs: list[str], device=device):
    if len(docs) == 0:
        return np.array([])
    inputs = tokenizer(
        [query]*len(docs),    
        docs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits  = outputs.logits.squeeze(-1) 
    probs   = torch.sigmoid(logits).cpu().numpy()
    return probs

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def topk_from_similarity_vector(similarities, k=3):
    sims = to_numpy(similarities).reshape(-1)
    if sims.size == 0:
        return []
    topk = np.argsort(-sims)[:k]
    return topk.tolist()

# def normalize(x):
#     x = to_numpy(x)
#     norms = np.linalg.norm(x, axis=-1, keepdims=True)
#     return x / (norms + 1e-12)


class ParagraphIn(BaseModel):
    idx: int
    title: str
    paragraph_text: str
    is_supporting: bool

# class SearchRequest(BaseModel):
#     paragraphs: List[ParagraphIn]
#     question: str
#     top_k: int = 3

class QuestionSubmission(BaseModel):
    # session_id: str
    question: str
    top_k: int = 3

class ParagraphHit(BaseModel):
    request_list_idx: int    # index in the request list
    paragraph_idx_field: int # paragraph['idx']
    title: str
    paragraph_text: str
    score: float

class SearchResponse(BaseModel):
    question: str
    top_hits: List[ParagraphHit]

class ParagraphSubmission(BaseModel):
    # session_id: str
    paragraphs: List[ParagraphIn]

@app.post("/submit_paragraphs")
async def submit_paragraphs(req: ParagraphSubmission):
    """Store paragraphs under a session ID (so different agents can contribute)."""
    if not req.paragraphs:
        raise HTTPException(status_code=400, detail="paragraphs are required.")
    # paragraph_store[req.session_id] = req.paragraphs
    paragraph_store.append(req.paragraphs)
    return {"message": f"Stored {len(req.paragraphs)} paragraphs as submission #{len(paragraph_store)}."}

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(req: QuestionSubmission):
    if not paragraph_store:
        raise HTTPException(status_code=404, detail=f"No paragraphs found.")
    
    if not req.question:
        raise HTTPException(status_code=400, detail="Question is required.")

    paragraphs = paragraph_store[-1]
    
    try:
        paragraph_texts = [p.paragraph_text for p in paragraphs]
        similarities = rerank_one(req.question, paragraph_texts, device=device)
        topk_idxs = topk_from_similarity_vector(similarities, k=req.top_k)

        hits = []
        for rank_idx in topk_idxs:
            p = paragraphs[rank_idx]
            hits.append(ParagraphHit(
                request_list_idx=rank_idx,
                paragraph_idx_field=p.idx,
                title=p.title,
                paragraph_text=p.paragraph_text,
                score=float(similarities[rank_idx])
            ))

        paragraph_store.pop()

        return SearchResponse(question=req.question, top_hits=hits)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))