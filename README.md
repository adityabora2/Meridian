# Meredian

**An Adaptive RAG System — Status: Currently Building**

A document Q&A system that decides *how* to retrieve before it retrieves — instead of running every query through the same fixed pipeline.

## The Problem

Most RAG systems follow a single path regardless of query complexity:

```
Question → Search → Retrieve Chunks → Generate Answer
```

- Simple questions get over-retrieved — noisy context, worse answers
- Complex multi-hop questions get under-retrieved — one search isn't enough, hallucinations follow

## The Approach

A lightweight router classifies each query's complexity *before* any retrieval happens, then sends it down one of three paths:

- **Mode 1 — Direct Answer**
  No retrieval needed. The LLM answers directly from its own knowledge.

- **Mode 2 — Single-Hop Retrieval**
  One vector search against the document store, followed by grounded generation with citations.

- **Mode 3 — Multi-Hop Retrieval + Self-Critique**
  Decomposes the question into sub-questions, retrieves evidence for each, chains it together, generates an answer and then verifies every claim against the retrieved evidence — looping back to search again if a claim isn't supported (capped at 3 iterations).

The routing decision is surfaced directly in the UI, so it's clear which mode handled each query.

## Stack

| Layer | Tool |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq API |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | FAISS (disk-persisted) |
| PDF Ingestion | PyMuPDF (512-token chunks, 50-token overlap) |
| Frontend | Streamlit |

This is a fully local, self-contained demo — no cloud services required to run it.

## Dataset

The system is tested against a small set of publicly available research papers, so the domain of the documents directly matches the reasoning the routing system is built to handle.

## Status

Currently building. Architecture and query-routing design are finalized; implementation is in progress.
