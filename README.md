# Hybrid RAG System with Multi-Intent Classification

A robust Retrieval-Augmented Generation (RAG) pipeline designed to handle diverse user queries by routing them through a custom intent classifier before retrieval.

## 🌟 Key Features
* **Hybrid Search:** Combines semantic search (SBERT) with keyword-based retrieval (BM25) for high-accuracy document fetching.
* **Intent Routing:** Uses a Logistic Regression classifier to categorize queries into **AI_QUERY, CODING, MATH, or CHITCHAT**, ensuring the LLM receives the right context.
* **Smart Chunking:** Implements text cleaning and overlapping word-based chunking for better context retention.
* **Local LLM Integration:** Optimized to work with local inference via Ollama.

## 📁 Project Structure
* `ingest.py`: Handles PDF processing and vector database (ChromaDB) storage.
* `train_intent.py`: Script to train the Logistic Regression model using `intent_data.csv`.
* `query.py`: The main entry point for the hybrid retrieval and RAG logic.
* `utils.py`: Shared helper functions for text preprocessing.

## 🚀 How to Run
1. Install dependencies: `pip install sentence-transformers chromadb rank_bm25 scikit-learn pandas`
2. Prepare your data: Place PDFs in the `docs/` folder.
3. Ingest: `python ingest.py`
4. Query: `python query.py`
