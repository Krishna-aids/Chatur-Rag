"""
main.py
-------
Vercel Serverless & Production Entrypoint for Chatur-RAG.

Exports:
    app         - FastAPI ASGI Application (Vercel Python Runtime)
    handler     - AWS Lambda / Vercel Serverless Handler alias
    application - WSGI/ASGI application alias
"""

import os
import sys
import tempfile
import uuid
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Set serverless friendly cache directories in /tmp
if os.getenv("VERCEL") == "1" or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    tmp_dir = tempfile.gettempdir()
    os.environ.setdefault("HF_HOME", os.path.join(tmp_dir, "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(tmp_dir, "huggingface"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(tmp_dir, "huggingface"))
    os.environ.setdefault("TORCH_HOME", os.path.join(tmp_dir, "torch"))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import CONFIG, GROQ_API_KEY
from utils.logger import get_logger

logger = get_logger("chatur_main")

# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Chatur-RAG",
    description="Adaptive Understanding & Retrieval Architecture (AURA-RAG)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Vercel entrypoint aliases
handler = app
application = app

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Lazy Pipeline Loader
# ---------------------------------------------------------------------------
_pipeline_instance = None


def get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        from core.pipeline import RAGPipeline
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., description="User question or query")
    session_id: Optional[str] = Field(default="web_session", description="Session identifier")


class QueryResponse(BaseModel):
    query_id: str
    query: str
    answer: str
    confidence: float
    grounded: bool
    sources: List[str]
    status: str


class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="Query ID being rated")
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="Answer that was generated")
    feedback: str = Field(..., description="User feedback or rating reason")
    rating: Optional[int] = Field(default=5, description="1 to 5 star rating")


class FeedbackResponse(BaseModel):
    query_id: str
    accepted: bool
    category: Optional[str] = None
    note: Optional[str] = None
    param_changes: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    documents: Optional[List[str]] = Field(default=None, description="List of raw document texts")
    chunk_size: int = Field(default=500, description="Chunk size in characters")
    chunk_overlap: int = Field(default=80, description="Chunk overlap in characters")


# ---------------------------------------------------------------------------
# Sample Documents for 1-Click Ingestion
# ---------------------------------------------------------------------------
SAMPLE_DOCUMENTS = [
    "Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system with a language model. The retrieval step fetches relevant documents from an external knowledge base, which are then passed as context to the generator to produce grounded answers.",
    "RAG reduces hallucinations by grounding the language model's output in retrieved external knowledge rather than relying solely on parametric memory baked in during training.",
    "The key components of a RAG system are: a document store, an embedding model to encode queries and documents, a retrieval mechanism, and a language model for answer generation.",
    "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It supports exact and approximate nearest neighbor search using methods like IVF (Inverted File Index) and HNSW.",
    "FAISS normalises vectors before indexing when using inner product (IP) indexes, which is equivalent to cosine similarity for normalised vectors. This makes it suitable for semantic search over sentence embeddings.",
    "FAISS stores vectors in memory and supports GPU acceleration. For production use, the IndexIVFFlat variant partitions the space into clusters, reducing search time by only scanning the nearest clusters.",
    "BM25 (Best Match 25) is a bag-of-words retrieval function that scores documents based on the query terms appearing in each document, adjusted by term frequency and inverse document frequency (TF-IDF).",
    "Hybrid search combines dense retrieval (semantic, using embeddings) with sparse retrieval (keyword-based, like BM25). Dense retrieval handles synonyms and paraphrases; BM25 handles exact term matches and rare proper nouns.",
    "Reciprocal Rank Fusion (RRF) is a rank aggregation algorithm that merges multiple ranked lists without requiring score normalisation. Each document's final score is sum(1/(k+rank)) across all lists, where k is typically 60.",
    "Cross-encoders process the query and candidate document together in a single forward pass through the model, enabling full attention between them. This produces more accurate relevance scores than bi-encoders that encode query and document independently.",
    "In a two-stage retrieval pipeline, a fast bi-encoder (or BM25) first retrieves a large candidate set, then a cross-encoder reranks the top candidates. This balances speed and accuracy.",
    "The cross-encoder model cross-encoder/ms-marco-MiniLM-L-6-v2 is trained on the MS MARCO passage ranking dataset and provides strong reranking quality at small model size.",
    "Sentence-transformers/all-MiniLM-L6-v2 is a compact 384-dimensional sentence embedding model. It is fast, runs on CPU, and performs well on semantic similarity tasks, making it suitable for real-time RAG pipelines.",
    "Embedding models map variable-length text to fixed-size dense vectors in a semantic space. Texts with similar meanings are mapped to nearby vectors, enabling similarity search.",
    "Groq provides an LLM inference API using their Language Processing Unit (LPU) hardware, achieving very high token throughput compared to GPU-based inference. This makes it practical to use large models like LLaMA 3 70B in latency-sensitive pipelines.",
    "LLaMA 3 8B is a lightweight open-source language model suitable for classification, intent detection, query rewriting, and structured output generation. LLaMA 3 70B is a larger model with stronger reasoning capabilities for complex generation tasks.",
    "Confidence evaluation in RAG involves checking whether every factual claim in the generated answer is directly supported by the retrieved context. This is different from general LLM confidence which is based on token probabilities.",
    "RAGAS is a framework for evaluating RAG pipelines using metrics like context precision, context recall, answer faithfulness, and answer relevance. These metrics can be computed without human annotation using LLM-based evaluation.",
    "Answer faithfulness measures the fraction of claims in the generated answer that are supported by the provided context. High faithfulness indicates low hallucination.",
    "ChromaDB is a vector database with built-in metadata filtering and collection management. In AURA-RAG it serves as the episodic memory store, persisting past interactions for similarity-based recall.",
    "Episodic memory in RAG systems stores past (query, answer, feedback) tuples. When a new query arrives, semantically similar past interactions are retrieved and injected into the context, enabling the system to learn from history.",
]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"])
@app.get("/api/health", tags=["System"])
def health_check():
    """Health check endpoint for Vercel deployment and monitoring."""
    api_key_set = bool(os.getenv("GROQ_API_KEY", "").strip() or GROQ_API_KEY)
    return {
        "status": "healthy",
        "service": "Chatur-RAG",
        "version": "1.0.0",
        "groq_api_key_configured": api_key_set,
        "environment": "vercel" if os.getenv("VERCEL") == "1" else "local",
    }


@app.get("/api/status", tags=["System"])
def system_status():
    """Returns real-time status of pipeline components, memory, and config."""
    api_key_set = bool(os.getenv("GROQ_API_KEY", "").strip() or GROQ_API_KEY)
    faiss_exists = os.path.exists("vectorstores/faiss_index/index.faiss")
    return {
        "status": "online",
        "groq_api_key_configured": api_key_set,
        "pipeline_loaded": _pipeline_instance is not None,
        "faiss_index_on_disk": faiss_exists,
        "config": {
            "top_k_semantic": CONFIG.retrieval.top_k_semantic,
            "top_k_keyword": CONFIG.retrieval.top_k_keyword,
            "top_k_rerank": CONFIG.retrieval.top_k_rerank,
            "ranking_threshold": CONFIG.ranking.threshold,
            "min_confidence": CONFIG.confidence.min_confidence,
            "memory_similarity_threshold": CONFIG.memory.similarity_threshold,
        },
    }


@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
def run_query(payload: QueryRequest):
    """
    Execute a full AURA-RAG query:
    1. Query rewrites & intent analysis (LLaMA 3 8B)
    2. Hybrid retrieval (FAISS dense + BM25 keyword + RRF fusion)
    3. Multi-stage ranking (Cross-Encoder + LLaMA 3 70B)
    4. Episodic memory lookup (ChromaDB)
    5. Answer generation (LLaMA 3 70B)
    6. Grounding & Faithfulness confidence check
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip() or GROQ_API_KEY
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="GROQ_API_KEY is not configured. Please set the GROQ_API_KEY environment variable in your Vercel Project Settings or .env file.",
        )

    try:
        pipeline = get_pipeline()
        result = pipeline.query(payload.query, session_id=payload.session_id)
        return QueryResponse(
            query_id=result.query_id,
            query=result.query,
            answer=result.answer,
            confidence=round(result.confidence, 3),
            grounded=result.grounded,
            sources=result.chunks_used,
            status="success" if result.grounded else "fallback",
        )
    except Exception as e:
        logger.error("query_execution_error", error=str(e), query=payload.query)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/api/feedback", response_model=FeedbackResponse, tags=["RAG"])
def submit_feedback(payload: FeedbackRequest):
    """
    Submit user feedback on an answer to trigger the self-improvement learning engine.
    Adapts top_k, ranking threshold, and confidence parameters dynamically.
    """
    try:
        pipeline = get_pipeline()
        res = pipeline.submit_feedback(
            query_id=payload.query_id,
            query=payload.query,
            answer=payload.answer,
            feedback=payload.feedback,
            rating=payload.rating,
        )
        return FeedbackResponse(
            query_id=res.query_id,
            accepted=res.accepted,
            category=res.category,
            note=res.note,
            param_changes=res.param_changes or {},
        )
    except Exception as e:
        logger.error("feedback_submission_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")


@app.post("/api/ingest/sample", tags=["Ingestion"])
def ingest_sample_corpus():
    """Ingest the 21 built-in sample AI/RAG documents into FAISS and Chroma."""
    from core.ingestion.pipeline import IngestionPipeline

    tmp_dir = tempfile.mkdtemp(prefix="chatur_sample_")
    try:
        for idx, doc_text in enumerate(SAMPLE_DOCUMENTS):
            fp = os.path.join(tmp_dir, f"sample_{idx:03d}.txt")
            with open(fp, "w", encoding="utf-8") as f:
                f.write(doc_text)

        faiss_dir = os.path.join(tempfile.gettempdir(), "faiss_index") if os.getenv("VERCEL") == "1" else "vectorstores/faiss_index"
        chroma_dir = os.path.join(tempfile.gettempdir(), "chroma_memory") if os.getenv("VERCEL") == "1" else "vectorstores/chroma_memory"

        ingestion = IngestionPipeline(
            source_path=tmp_dir,
            chunk_size=500,
            chunk_overlap=80,
            faiss_path=faiss_dir,
            chroma_path=chroma_dir,
        )
        ingestion.run()

        # Re-initialize pipeline if already loaded to pick up new index
        global _pipeline_instance
        _pipeline_instance = None

        return {
            "status": "completed",
            "documents_ingested": len(SAMPLE_DOCUMENTS),
            "message": "Sample documents successfully indexed into FAISS and Chroma memory.",
        }
    except Exception as e:
        logger.error("sample_ingestion_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


# ---------------------------------------------------------------------------
# ChatGPT-Style Rich Web User Interface
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, tags=["Web UI"])
def render_web_ui():
    """Serves a modern, ChatGPT-style web interface for Chatur-RAG."""
    return HTMLResponse(content=WEB_UI_HTML)


WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Chatur-RAG | Self-Improving Adaptive RAG</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #090a0f;
      --bg-surface: #10121a;
      --bg-surface-elevated: #171a26;
      --border-subtle: #232738;
      --border-accent: rgba(99, 102, 241, 0.4);
      --primary: #6366f1;
      --primary-hover: #4f46e5;
      --primary-glow: rgba(99, 102, 241, 0.25);
      --accent: #06b6d4;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-faint: #64748b;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
      --sidebar-width: 320px;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }

    body {
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      height: 100vh;
      overflow: hidden;
      display: flex;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
      width: 6px;
      height: 6px;
    }
    ::-webkit-scrollbar-track {
      background: transparent;
    }
    ::-webkit-scrollbar-thumb {
      background: var(--border-subtle);
      border-radius: 9999px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #3b4259;
    }

    /* Layout */
    .app-container {
      display: flex;
      width: 100vw;
      height: 100vh;
      position: relative;
    }

    /* Sidebar */
    .sidebar {
      width: var(--sidebar-width);
      background: var(--bg-surface);
      border-right: 1px solid var(--border-subtle);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
      z-index: 20;
      transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .sidebar-header {
      padding: 20px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-icon {
      width: 38px;
      height: 38px;
      background: linear-gradient(135deg, var(--primary), var(--accent));
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 18px;
      color: white;
      box-shadow: 0 4px 15px var(--primary-glow);
    }

    .brand-title {
      font-weight: 700;
      font-size: 17px;
      letter-spacing: -0.02em;
      background: linear-gradient(to right, #ffffff, #cbd5e1);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .brand-subtitle {
      font-size: 11px;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .sidebar-content {
      padding: 18px;
      flex: 1;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .section-title {
      font-size: 11px;
      font-weight: 600;
      color: var(--text-faint);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }

    .status-card {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .status-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
    }

    .pill {
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 9999px;
      display: inline-flex;
      align-items: center;
      gap: 5px;
    }

    .pill.online {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .pill.warning {
      background: rgba(245, 158, 11, 0.15);
      color: #fbbf24;
      border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .pill-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: currentColor;
    }

    .quick-btn {
      width: 100%;
      text-align: left;
      padding: 10px 12px;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      color: var(--text-muted);
      font-size: 12px;
      font-family: inherit;
      cursor: pointer;
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .quick-btn:hover {
      background: rgba(99, 102, 241, 0.1);
      border-color: var(--border-accent);
      color: var(--text-main);
      transform: translateY(-1px);
    }

    .sidebar-footer {
      padding: 16px 20px;
      border-top: 1px solid var(--border-subtle);
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      color: var(--text-faint);
    }

    .sidebar-footer a {
      color: var(--primary);
      text-decoration: none;
    }

    /* Main Chat Container */
    .main-chat {
      flex: 1;
      display: flex;
      flex-direction: column;
      height: 100vh;
      background: radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.05) 0%, transparent 60%);
      position: relative;
    }

    .top-nav {
      height: 64px;
      border-bottom: 1px solid var(--border-subtle);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 28px;
      background: rgba(16, 18, 26, 0.7);
      backdrop-filter: blur(12px);
    }

    .top-nav-title {
      font-weight: 600;
      font-size: 15px;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .chat-messages {
      flex: 1;
      overflow-y: auto;
      padding: 24px 28px;
      display: flex;
      flex-direction: column;
      gap: 24px;
      max-width: 900px;
      margin: 0 auto;
      width: 100%;
    }

    /* Welcome / Empty state */
    .empty-state {
      margin: auto;
      text-align: center;
      max-width: 580px;
      padding: 40px 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
    }

    .empty-icon {
      width: 56px;
      height: 56px;
      background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(6, 182, 212, 0.2));
      border: 1px solid var(--border-accent);
      border-radius: 16px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 24px;
      box-shadow: 0 8px 30px var(--primary-glow);
    }

    .empty-title {
      font-size: 24px;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .empty-desc {
      color: var(--text-muted);
      font-size: 14px;
      line-height: 1.6;
    }

    .prompt-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      width: 100%;
      margin-top: 12px;
    }

    .prompt-card {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 14px;
      text-align: left;
      cursor: pointer;
      transition: all 0.2s;
    }

    .prompt-card:hover {
      border-color: var(--primary);
      transform: translateY(-2px);
      box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
    }

    .prompt-card-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-main);
      margin-bottom: 4px;
    }

    .prompt-card-sub {
      font-size: 11px;
      color: var(--text-muted);
    }

    /* Message Bubbles */
    .message-row {
      display: flex;
      gap: 14px;
      animation: fadeIn 0.3s ease-out;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .avatar {
      width: 34px;
      height: 34px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 600;
      font-size: 13px;
      flex-shrink: 0;
    }

    .user-avatar {
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      color: white;
    }

    .ai-avatar {
      background: linear-gradient(135deg, var(--primary), var(--accent));
      color: white;
      box-shadow: 0 4px 12px var(--primary-glow);
    }

    .message-body {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }

    .user-bubble {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 12px 18px;
      font-size: 14px;
      line-height: 1.6;
      width: fit-content;
      max-width: 85%;
      margin-left: auto;
    }

    .ai-bubble {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 18px 20px;
      font-size: 14px;
      line-height: 1.7;
      color: #e2e8f0;
    }

    .badge-bar {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 10px;
      padding-top: 12px;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
      flex-wrap: wrap;
    }

    .confidence-meter {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      font-weight: 600;
    }

    .confidence-bar-bg {
      width: 70px;
      height: 6px;
      background: var(--bg-surface-elevated);
      border-radius: 9999px;
      overflow: hidden;
    }

    .confidence-bar-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--success));
      border-radius: 9999px;
    }

    .sources-toggle {
      font-size: 12px;
      color: var(--text-muted);
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px 8px;
      border-radius: 6px;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      transition: all 0.2s;
    }

    .sources-toggle:hover {
      color: var(--text-main);
      border-color: var(--border-accent);
    }

    .sources-list {
      margin-top: 10px;
      padding: 12px;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
    }

    .source-item {
      padding: 8px 10px;
      background: rgba(0, 0, 0, 0.2);
      border-left: 2px solid var(--primary);
      border-radius: 4px;
      color: #cbd5e1;
      white-space: pre-wrap;
    }

    .feedback-group {
      margin-left: auto;
      display: flex;
      gap: 6px;
    }

    .feedback-btn {
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      border-radius: 6px;
      padding: 4px 8px;
      font-size: 12px;
      cursor: pointer;
      transition: all 0.2s;
    }

    .feedback-btn:hover {
      color: var(--text-main);
      border-color: var(--primary);
    }

    /* Input Bar */
    .input-container {
      padding: 20px 28px 24px;
      max-width: 900px;
      margin: 0 auto;
      width: 100%;
    }

    .input-box {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      padding: 12px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
      transition: border-color 0.2s, box-shadow 0.2s;
    }

    .input-box:focus-within {
      border-color: var(--primary);
      box-shadow: 0 8px 30px var(--primary-glow);
    }

    .query-textarea {
      background: transparent;
      border: none;
      color: var(--text-main);
      font-family: inherit;
      font-size: 14px;
      resize: none;
      outline: none;
      min-height: 44px;
      max-height: 160px;
      line-height: 1.5;
    }

    .query-textarea::placeholder {
      color: var(--text-faint);
    }

    .input-actions {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .input-hint {
      font-size: 11px;
      color: var(--text-faint);
    }

    .send-btn {
      background: linear-gradient(135deg, var(--primary), var(--primary-hover));
      color: white;
      border: none;
      border-radius: 10px;
      padding: 8px 18px;
      font-weight: 600;
      font-size: 13px;
      font-family: inherit;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
      box-shadow: 0 4px 12px var(--primary-glow);
    }

    .send-btn:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 6px 18px rgba(99, 102, 241, 0.4);
    }

    .send-btn:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    /* Modal */
    .modal-backdrop {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(4px);
      z-index: 100;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }

    .modal-card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 16px;
      max-width: 480px;
      width: 100%;
      padding: 24px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .modal-title {
      font-size: 17px;
      font-weight: 700;
    }

    .modal-input {
      width: 100%;
      background: var(--bg-surface-elevated);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      color: var(--text-main);
      font-family: inherit;
      font-size: 13px;
      padding: 10px;
      outline: none;
    }

    .modal-buttons {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      margin-top: 8px;
    }

    .btn-secondary {
      background: transparent;
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      border-radius: 8px;
      padding: 8px 14px;
      cursor: pointer;
    }

    .btn-primary {
      background: var(--primary);
      border: none;
      color: white;
      border-radius: 8px;
      padding: 8px 14px;
      font-weight: 600;
      cursor: pointer;
    }

    /* Mobile toggle */
    .mobile-menu-btn {
      display: none;
      background: transparent;
      border: none;
      color: var(--text-main);
      font-size: 20px;
      cursor: pointer;
    }

    @media (max-width: 768px) {
      .sidebar {
        position: absolute;
        height: 100%;
        transform: translateX(-100%);
      }
      .sidebar.open {
        transform: translateX(0);
      }
      .mobile-menu-btn {
        display: block;
      }
      .prompt-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="app-container">
    <!-- Sidebar -->
    <aside class="sidebar" id="sidebar">
      <div class="sidebar-header">
        <div class="brand-icon">◈</div>
        <div>
          <div class="brand-title">Chatur-RAG</div>
          <div class="brand-subtitle">Adaptive Retrieval Architecture</div>
        </div>
      </div>

      <div class="sidebar-content">
        <!-- Status Card -->
        <div>
          <div class="section-title">Deployment Status</div>
          <div class="status-card" id="statusCard">
            <div class="status-row">
              <span style="color: var(--text-muted);">Runtime:</span>
              <span class="pill online"><span class="pill-dot"></span> Vercel Ready</span>
            </div>
            <div class="status-row">
              <span style="color: var(--text-muted);">Groq API:</span>
              <span class="pill" id="groqStatusPill"><span class="pill-dot"></span> Checking...</span>
            </div>
            <div class="status-row">
              <span style="color: var(--text-muted);">Vector Store:</span>
              <span class="pill online"><span class="pill-dot"></span> FAISS + BM25</span>
            </div>
          </div>
        </div>

        <!-- Quick Ingestion -->
        <div>
          <div class="section-title">Knowledge Base</div>
          <button class="quick-btn" id="btnIngestSample" onclick="ingestSampleDocs()">
            <span>⚡</span>
            <span>Re-Index Sample Corpus (21 Docs)</span>
          </button>
        </div>

        <!-- System Parameters -->
        <div>
          <div class="section-title">Learning Engine Parameters</div>
          <div class="status-card" id="paramsCard">
            <div class="status-row">
              <span style="color: var(--text-muted);">Semantic Top-K:</span>
              <span id="paramTopKSemantic" style="font-family: monospace;">20</span>
            </div>
            <div class="status-row">
              <span style="color: var(--text-muted);">BM25 Top-K:</span>
              <span id="paramTopKKeyword" style="font-family: monospace;">20</span>
            </div>
            <div class="status-row">
              <span style="color: var(--text-muted);">Ranking Threshold:</span>
              <span id="paramRankingThreshold" style="font-family: monospace;">6.0</span>
            </div>
            <div class="status-row">
              <span style="color: var(--text-muted);">Min Confidence:</span>
              <span id="paramMinConfidence" style="font-family: monospace;">0.70</span>
            </div>
          </div>
        </div>

        <!-- Sample Questions -->
        <div>
          <div class="section-title">Sample Prompts</div>
          <div style="display: flex; flex-direction: column; gap: 6px;">
            <button class="quick-btn" onclick="sendSample('What is retrieval-augmented generation and how does it reduce hallucinations?')">
              <span>✦</span> What is RAG?
            </button>
            <button class="quick-btn" onclick="sendSample('How does FAISS perform approximate nearest neighbor search?')">
              <span>✦</span> FAISS similarity search
            </button>
            <button class="quick-btn" onclick="sendSample('What is the difference between BM25 and dense retrieval, and when should I use hybrid search?')">
              <span>✦</span> BM25 vs Dense Retrieval
            </button>
            <button class="quick-btn" onclick="sendSample('Why use a cross-encoder for reranking?')">
              <span>✦</span> Cross-encoder reranking
            </button>
            <button class="quick-btn" onclick="sendSample('How does the AURA-RAG memory system work with ChromaDB?')">
              <span>✦</span> ChromaDB memory loop
            </button>
          </div>
        </div>
      </div>

      <div class="sidebar-footer">
        <span>Chatur-RAG v1.0.0</span>
        <a href="/docs" target="_blank">Swagger API ↗</a>
      </div>
    </aside>

    <!-- Main Chat Content -->
    <main class="main-chat">
      <!-- Top Navigation -->
      <header class="top-nav">
        <div style="display: flex; align-items: center; gap: 12px;">
          <button class="mobile-menu-btn" onclick="toggleSidebar()">☰</button>
          <div class="top-nav-title">
            <span>Chat Session</span>
            <span class="pill online" style="font-size: 10px;">FastAPI + Vercel</span>
          </div>
        </div>
        <div style="display: flex; gap: 8px;">
          <button class="quick-btn" style="width: auto; padding: 6px 12px;" onclick="clearChat()">Clear Chat</button>
        </div>
      </header>

      <!-- Messages Area -->
      <div class="chat-messages" id="chatMessages">
        <div class="empty-state" id="emptyState">
          <div class="empty-icon">◈</div>
          <h1 class="empty-title">Chatur-RAG</h1>
          <p class="empty-desc">
            A self-improving Retrieval-Augmented Generation system. Combines FAISS semantic search, BM25 keyword matching, Reciprocal Rank Fusion, Cross-Encoder reranking, and ChromaDB episodic memory.
          </p>
          <div class="prompt-grid">
            <div class="prompt-card" onclick="sendSample('What is retrieval-augmented generation and how does it reduce hallucinations?')">
              <div class="prompt-card-title">What is RAG?</div>
              <div class="prompt-card-sub">Explain how external knowledge grounds responses</div>
            </div>
            <div class="prompt-card" onclick="sendSample('What is the difference between BM25 and dense retrieval, and when should I use hybrid search?')">
              <div class="prompt-card-title">Hybrid Search</div>
              <div class="prompt-card-sub">Compare sparse BM25 vs dense embeddings</div>
            </div>
            <div class="prompt-card" onclick="sendSample('Why use a cross-encoder for reranking?')">
              <div class="prompt-card-title">Cross-Encoder Reranking</div>
              <div class="prompt-card-sub">Understand two-stage retrieval benefits</div>
            </div>
            <div class="prompt-card" onclick="sendSample('How does the AURA-RAG memory system work with ChromaDB?')">
              <div class="prompt-card-title">Episodic Memory</div>
              <div class="prompt-card-sub">How feedback loops continuously improve accuracy</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Input Bar -->
      <footer class="input-container">
        <div class="input-box">
          <textarea
            id="queryInput"
            class="query-textarea"
            placeholder="Ask anything about RAG, FAISS, BM25, embeddings, or your indexed docs..."
            rows="1"
            onkeydown="handleKey(event)"
          ></textarea>
          <div class="input-actions">
            <span class="input-hint">Shift + Enter for new line • Powered by Groq LLaMA 3</span>
            <button id="sendBtn" class="send-btn" onclick="submitQuery()">
              <span>Ask Chatur</span>
              <span>→</span>
            </button>
          </div>
        </div>
      </footer>
    </main>
  </div>

  <!-- Feedback Modal -->
  <div class="modal-backdrop" id="feedbackModal">
    <div class="modal-card">
      <div class="modal-title">Provide Feedback to Learning Engine</div>
      <p style="font-size: 13px; color: var(--text-muted);">
        Chatur-RAG learns from your feedback to automatically calibrate retrieval depth, reranking thresholds, and confidence bounds.
      </p>
      <div>
        <label style="font-size: 12px; color: var(--text-faint); display: block; margin-bottom: 6px;">Feedback Details</label>
        <textarea id="feedbackText" class="modal-input" rows="3" placeholder="e.g. Missing technical details about BM25 scoring..."></textarea>
      </div>
      <div class="modal-buttons">
        <button class="btn-secondary" onclick="closeFeedbackModal()">Cancel</button>
        <button class="btn-primary" onclick="submitFeedbackFromModal()">Submit & Adapt</button>
      </div>
    </div>
  </div>

  <script>
    let currentFeedbackData = null;

    // Check system status on page load
    async function checkStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        const pill = document.getElementById('groqStatusPill');
        if (data.groq_api_key_configured) {
          pill.className = 'pill online';
          pill.innerHTML = '<span class="pill-dot"></span> Groq Connected';
        } else {
          pill.className = 'pill warning';
          pill.innerHTML = '<span class="pill-dot"></span> Set GROQ_API_KEY';
        }

        if (data.config) {
          document.getElementById('paramTopKSemantic').innerText = data.config.top_k_semantic;
          document.getElementById('paramTopKKeyword').innerText = data.config.top_k_keyword;
          document.getElementById('paramRankingThreshold').innerText = data.config.ranking_threshold;
          document.getElementById('paramMinConfidence').innerText = data.config.min_confidence;
        }
      } catch (e) {
        console.error('Status check error:', e);
      }
    }
    checkStatus();

    function toggleSidebar() {
      document.getElementById('sidebar').classList.toggle('open');
    }

    function handleKey(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitQuery();
      }
    }

    function sendSample(text) {
      document.getElementById('queryInput').value = text;
      submitQuery();
    }

    function clearChat() {
      const container = document.getElementById('chatMessages');
      container.innerHTML = `
        <div class="empty-state" id="emptyState">
          <div class="empty-icon">◈</div>
          <h1 class="empty-title">Chatur-RAG</h1>
          <p class="empty-desc">
            A self-improving Retrieval-Augmented Generation system. Combines FAISS semantic search, BM25 keyword matching, Reciprocal Rank Fusion, Cross-Encoder reranking, and ChromaDB episodic memory.
          </p>
        </div>
      `;
    }

    async function submitQuery() {
      const input = document.getElementById('queryInput');
      const query = input.value.trim();
      if (!query) return;

      const empty = document.getElementById('emptyState');
      if (empty) empty.remove();

      const messages = document.getElementById('chatMessages');

      // Append User message
      const userRow = document.createElement('div');
      userRow.className = 'message-row';
      userRow.innerHTML = `
        <div class="avatar user-avatar">U</div>
        <div class="message-body">
          <div class="user-bubble">${escapeHtml(query)}</div>
        </div>
      `;
      messages.appendChild(userRow);

      input.value = '';
      input.disabled = true;
      document.getElementById('sendBtn').disabled = true;

      // Append AI placeholder
      const aiRow = document.createElement('div');
      aiRow.className = 'message-row';
      const aiMsgId = 'ai-msg-' + Date.now();
      aiRow.innerHTML = `
        <div class="avatar ai-avatar">◈</div>
        <div class="message-body" id="${aiMsgId}">
          <div class="ai-bubble" style="display: flex; align-items: center; gap: 8px;">
            <span class="pill-dot" style="background: var(--primary); animation: pulse 1s infinite;"></span>
            <span style="color: var(--text-muted); font-size: 13px;">Reasoning through hybrid retrieval & cross-encoder ranking...</span>
          </div>
        </div>
      `;
      messages.appendChild(aiRow);
      messages.scrollTop = messages.scrollHeight;

      try {
        const res = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: query, session_id: 'web_session' })
        });

        const data = await res.json();
        const container = document.getElementById(aiMsgId);

        if (!res.ok) {
          container.innerHTML = `
            <div class="ai-bubble" style="border-color: var(--danger);">
              <span style="color: #f87171; font-weight: 600;">Error:</span> ${escapeHtml(data.detail || 'An unexpected error occurred.')}
            </div>
          `;
          return;
        }

        const confidencePct = Math.round((data.confidence || 0) * 100);
        const sourcesCount = data.sources ? data.sources.length : 0;
        const groundedBadge = data.grounded
          ? '<span class="pill online"><span class="pill-dot"></span> Grounded in Context</span>'
          : '<span class="pill warning"><span class="pill-dot"></span> Low Confidence Fallback</span>';

        let sourcesHtml = '';
        if (sourcesCount > 0) {
          sourcesHtml = `
            <div class="sources-toggle" onclick="toggleSources('${aiMsgId}-sources')">
              <span>📄 Sources Used (${sourcesCount})</span>
              <span>▾</span>
            </div>
            <div class="sources-list" id="${aiMsgId}-sources" style="display: none;">
              ${data.sources.map((src, i) => `<div class="source-item"><strong>Chunk ${i+1}:</strong> ${escapeHtml(src)}</div>`).join('')}
            </div>
          `;
        }

        container.innerHTML = `
          <div class="ai-bubble">
            <div style="white-space: pre-wrap;">${escapeHtml(data.answer)}</div>
            <div class="badge-bar">
              <div class="confidence-meter">
                <span style="color: var(--text-muted);">Confidence:</span>
                <div class="confidence-bar-bg">
                  <div class="confidence-bar-fill" style="width: ${confidencePct}%;"></div>
                </div>
                <span>${confidencePct}%</span>
              </div>
              ${groundedBadge}
              ${sourcesHtml}
              <div class="feedback-group">
                <button class="feedback-btn" title="Helpful answer" onclick="openFeedback('${data.query_id}', '${escapeHtml(query)}', '${escapeHtml(data.answer)}', 'helpful')">👍 Helpful</button>
                <button class="feedback-btn" title="Needs improvement" onclick="openFeedback('${data.query_id}', '${escapeHtml(query)}', '${escapeHtml(data.answer)}', 'not_helpful')">👎 Improve</button>
              </div>
            </div>
          </div>
        `;
      } catch (err) {
        const container = document.getElementById(aiMsgId);
        container.innerHTML = `
          <div class="ai-bubble" style="border-color: var(--danger);">
            <span style="color: #f87171; font-weight: 600;">Network Error:</span> Could not connect to API server.
          </div>
        `;
      } finally {
        input.disabled = false;
        document.getElementById('sendBtn').disabled = false;
        input.focus();
        messages.scrollTop = messages.scrollHeight;
      }
    }

    function toggleSources(id) {
      const el = document.getElementById(id);
      if (el) {
        el.style.display = el.style.display === 'none' ? 'flex' : 'none';
      }
    }

    function openFeedback(queryId, query, answer, type) {
      currentFeedbackData = { query_id: queryId, query: query, answer: answer, feedback_type: type };
      document.getElementById('feedbackText').value = type === 'helpful' ? 'Clear and grounded answer.' : 'Could be improved.';
      document.getElementById('feedbackModal').style.display = 'flex';
    }

    function closeFeedbackModal() {
      document.getElementById('feedbackModal').style.display = 'none';
      currentFeedbackData = null;
    }

    async function submitFeedbackFromModal() {
      if (!currentFeedbackData) return;
      const comment = document.getElementById('feedbackText').value.trim();
      const payload = {
        query_id: currentFeedbackData.query_id,
        query: currentFeedbackData.query,
        answer: currentFeedbackData.answer,
        feedback: `[${currentFeedbackData.feedback_type}] ${comment}`,
        rating: currentFeedbackData.feedback_type === 'helpful' ? 5 : 2
      };

      try {
        const res = await fetch('/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        alert(`Learning Engine feedback received!\nCategory: ${data.category || 'general'}\nEngine adapted parameters.`);
        checkStatus();
      } catch (e) {
        alert('Failed to submit feedback.');
      } finally {
        closeFeedbackModal();
      }
    }

    async function ingestSampleDocs() {
      const btn = document.getElementById('btnIngestSample');
      btn.disabled = true;
      btn.innerHTML = '<span>⏳</span><span>Indexing...</span>';
      try {
        const res = await fetch('/api/ingest/sample', { method: 'POST' });
        const data = await res.json();
        if (res.ok) {
          alert('Successfully indexed 21 sample documents into FAISS and Chroma memory!');
        } else {
          alert('Ingestion error: ' + (data.detail || 'Failed'));
        }
      } catch (e) {
        alert('Ingestion request failed.');
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>⚡</span><span>Re-Index Sample Corpus (21 Docs)</span>';
        checkStatus();
      }
    }

    function escapeHtml(text) {
      if (!text) return '';
      return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Local Dev Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\nStarting Chatur-RAG FastAPI & Web UI at http://localhost:8000 ...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
