#!/usr/bin/env python3
# query.py - modular RAG query with Option-B intent classifier support

import os
import time
import logging
import json
import re
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import requests
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings
from sklearn.preprocessing import LabelEncoder
import joblib

# Optional cross-encoder (better rerank) - install cross-encoder to use
try:
    from cross_encoder import CrossEncoder
    HAS_CE = True
except Exception:
    HAS_CE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------- Config ----------
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "docs"          # must match your ingest.py
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
INTENT_CLASSIFIER_PATH = "intent_classifier.joblib"   # produced by training script
LABEL_ENCODER_PATH = "intent_label_encoder.joblib"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # optional cross-encoder

# LLM endpoint (Ollama-compatible). Use your Ollama model name (my-mistral).
LLM_API_URL = "http://127.0.0.1:11434/api/chat"
LLM_MODEL_NAME = "my-mistral"

# Retrieval knobs
TOP_K_DENSE = 40
TOP_K_BM25 = 40
UNION_K = 60
MMR_K = 12
FINAL_K = 4

# Confidence knobs for adaptive decision
MIN_TOP_SCORE = 0.20
DELTA_TOP2 = 0.05
MEAN_GAP_FACTOR = 0.25

# Post-filter alias map (kept small; classifier will largely remove fragility)
AI_ALIAS_MAP = {
    "lora": {"lora", "low rank adaptation", "low-rank adaptation", "lowrank"},
    "qlora": {"qlora", "quantized lora", "quantized low rank"},
    "rag": {"rag", "retrieval augmented generation", "retrieval-augmented generation"},
}

# ---------- Helpers ----------
def now_s(): return time.time()

def safe_json_from_response(text: str) -> Any:
    """Try to robustly parse JSON from possible noisy LLM output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # attempt to find a JSON object inside the text
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        # fallback: return raw text
        return {"message": {"content": text.strip()}}

def call_local_llm(prompt: str, max_tokens: int = 512) -> str:
    """Call Ollama-like endpoint; parse robustly."""
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    try:
        r = requests.post(LLM_API_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = safe_json_from_response(r.text)
        # try different shapes
        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], dict) and "content" in data["message"]:
                return data["message"]["content"].strip()
            if "choices" in data and data["choices"]:
                first = data["choices"][0]
                if "message" in first and "content" in first["message"]:
                    return first["message"]["content"].strip()
                if "text" in first:
                    return first["text"].strip()
        # fallback
        return str(data)
    except Exception as e:
        logging.error("LLM call failed: %s", e)
        return "⚠️ Error: could not generate response from local LLM."

# ---------- Intent classifier (Option B) ----------
def load_intent_classifier(path: str, label_encoder_path: str):
    if os.path.exists(path) and os.path.exists(label_encoder_path):
        logging.info("🔎 Loading intent classifier from %s", path)
        clf = joblib.load(path)
        le = joblib.load(label_encoder_path)
        return clf, le
    logging.warning("⚠️ Intent classifier not found at %s — will use prototype fallback until you train one.", path)
    return None, None

# Prototype fallback (only used if classifier absent) - few-shot prototypes using SBERT
PROTOTYPES = {
    "AI": [
        "What is retrieval augmented generation (RAG)?",
        "Explain transformer architecture",
        "What is LoRA fine tuning?"
    ],
    "CODING": [
        "Write a python function to reverse a linked list",
        "Show me Java code for BFS",
        "How to implement quicksort in Python?"
    ],
    "GENERAL": [
        "What is health?",
        "What is the capital of France?",
        "How do I boil an egg?"
    ],
    "CHITCHAT": [
        "You are doing great",
        "Thanks, you're awesome",
        "Good morning"
    ]
}

def prototype_intent(embedder, query: str) -> Tuple[str, float]:
    """Simple embedding nearest-prototype classifier (fallback)."""
    q_emb = embedder.encode(query, convert_to_numpy=True)
    best_label, best_sim = None, -1.0
    for label, examples in PROTOTYPES.items():
        ex_embs = embedder.encode(examples, convert_to_numpy=True)
        sim = np.max(util.cos_sim(q_emb, ex_embs).cpu().numpy())
        if sim > best_sim:
            best_sim = float(sim)
            best_label = label
    return best_label, best_sim

# ---------- Utility: post-filter keywords ----------
def extract_keywords(q: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9\-]+", q.lower())
    kws = [t for t in toks if len(t) >= 2]
    return list(dict.fromkeys(kws))

def required_aliases_present(query: str, texts: List[str]) -> bool:
    q_low = query.lower()
    required = set()
    for aliases in AI_ALIAS_MAP.values():
        if any(a in q_low for a in aliases):
            required |= aliases
    if not required:
        return True
    alltxt = " ".join(texts).lower()
    return any(a in alltxt for a in required)

# ---------- Retrieval: hybrid + MMR + optional cross-encoder rerank ----------
def build_bm25_index(docs: List[str]) -> Optional[BM25Okapi]:
    if not docs:
        return None
    tokenized = [d.split() for d in docs]
    return BM25Okapi(tokenized)

def mmr_select(query_emb: np.ndarray, candidate_embs: np.ndarray, candidate_idxs: List[int], k: int = 8, lambda_param: float = 0.7) -> List[int]:
    """
    Return indices (into candidate_idxs list) of selected items (MMR).
    candidate_embs: numpy array shape (N,D)
    candidate_idxs: list of ids (same length)
    """
    if len(candidate_idxs) <= k:
        return list(range(len(candidate_idxs)))
    sims = (candidate_embs @ query_emb.reshape(-1,1)).ravel()  # dot product (if embeddings normalized, it's cos)
    selected = []
    selected_idxs = []
    pool = set(range(len(candidate_idxs)))
    # pick highest sim first
    first = int(np.argmax(sims))
    selected.append(first)
    selected_idxs.append(first)
    pool.remove(first)
    while len(selected) < k and pool:
        mmr_scores = {}
        for i in pool:
            rel = sims[i]
            div = max([float(np.dot(candidate_embs[i], candidate_embs[j])) for j in selected_idxs]) if selected_idxs else 0.0
            mmr_scores[i] = lambda_param * rel - (1 - lambda_param) * div
        pick = max(mmr_scores.items(), key=lambda x: x[1])[0]
        selected.append(pick)
        selected_idxs.append(pick)
        pool.remove(pick)
    return selected

# ---------- Main pipeline ----------
def main():
    logging.info("🧠 Loading encoders...")
    embedder = SentenceTransformer(EMBED_MODEL)

    # load intent classifier (Option-B). Expect a sklearn-like classifier that accepts embeddings.
    clf, label_enc = load_intent_classifier(INTENT_CLASSIFIER_PATH, LABEL_ENCODER_PATH)

    # connect to chroma
    logging.info("🗂️ Loading ChromaDB collection...")
    client = chromadb.PersistentClient(path=CHROMA_PATH, settings=Settings(anonymized_telemetry=True))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        logging.error("Chroma collection '%s' not found: %s", COLLECTION_NAME, e)
        logging.info("Please run ingest.py to populate the collection.")
        return

    # hydrate docs + metas for BM25
    res = collection.get(include=["documents", "metadatas"])
    docs = res.get("documents", []) or []
    metas = res.get("metadatas", []) or []
    if not docs:
        logging.warning("No documents found in Chroma collection.")
    bm25 = build_bm25_index(docs)

    # optional cross-encoder
    reranker = None
    if HAS_CE:
        try:
            logging.info("Loading cross-encoder reranker...")
            reranker = CrossEncoder(RERANKER_MODEL)
        except Exception as e:
            logging.warning("Could not load cross-encoder: %s", e)
            reranker = None

    # REPL loop
    while True:
        try:
            query = input("Enter query (or 'exit'): ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            break

        t0 = now_s()

        # --- Intent detection ---
        if clf is not None and label_enc is not None:
            # compute embedding and call classifier (classifier expects embeddings)
            q_emb = embedder.encode(query, convert_to_numpy=True)
            try:
                pred = clf.predict([q_emb])[0]
                intent = pred if isinstance(pred, str) else label_enc.inverse_transform([pred])[0]
                intent_conf = None
            except Exception as e:
                logging.warning("Classifier predict error: %s - falling back to prototype", e)
                intent, intent_conf = prototype_intent(embedder, query)
        else:
            intent, intent_conf = prototype_intent(embedder, query)

        logging.info("📂 Intent: %s%s", intent, (f" (sim={intent_conf:.3f})" if intent_conf is not None else ""))

        # chit-chat handling: keep it short and friendly
        if intent == "CHITCHAT":
            # short set of canned responses
            lc = query.lower()
            if any(greet in lc for greet in ("hi", "hello", "hey")):
                resp = "Hi — how can I help with your research or questions today?"
            elif any(kw in lc for kw in ("thank", "thanks", "great", "nice", "good job", "well done", "you are doing")):
                resp = "Thanks — glad it's helping! What would you like next?"
            else:
                resp = "🙂 I'm here to help — ask me about your documents, AI topics, or coding."
            print("\n💡 Answer:\n", resp)
            print("\n📂 Source: none (chit-chat)")
            print(f"\n⏱️ Took {now_s()-t0:.2f}s")
            continue

        # Build expanded query for retrieval (alias expansion)
        q_expanded = query
        # retrieve dense top from Chroma
        q_emb = embedder.encode(query, convert_to_numpy=True)
        dense_res = collection.query(query_embeddings=[q_emb.tolist()], n_results=TOP_K_DENSE, include=["documents", "metadatas", "distances"])
        dense_docs = dense_res["documents"][0]
        dense_metas = dense_res["metadatas"][0]
        dense_ids = dense_res["ids"][0]
        dense_dists = dense_res["distances"][0]  # smaller distance => better (Chroma)

        # sparse BM25
        sparse_docs = []
        sparse_metas = []
        sparse_ids = []
        sparse_scores = []
        if bm25:
            b_scores = bm25.get_scores(query.split())
            top_idx = np.argsort(b_scores)[::-1][:TOP_K_BM25]
            for idx in top_idx:
                sparse_docs.append(docs[idx])
                sparse_metas.append(metas[idx] if idx < len(metas) else {})
                sparse_ids.append(dense_ids[idx] if idx < len(dense_ids) else str(idx))
                sparse_scores.append(float(b_scores[idx]))

        # merge pool, choose best score per id (convert dense distances to similarity)
        pool: Dict[str, Dict[str, Any]] = {}
        # from dense
        for doc, md, did, dist in zip(dense_docs, dense_metas, dense_ids, dense_dists):
            score = -float(dist)  # convert distance to similarity (approx)
            if did not in pool or score > pool[did]["score"]:
                pool[did] = {"text": doc or "", "meta": md or {}, "score": score}
        # from sparse
        for doc, md, did, s in zip(sparse_docs, sparse_metas, sparse_ids, sparse_scores):
            if did not in pool or s > pool[did]["score"]:
                pool[did] = {"text": doc or "", "meta": md or {}, "score": s}

        # limit to top UNION_K by pool score
        pool_items = sorted(pool.items(), key=lambda kv: kv[1]["score"], reverse=True)[:UNION_K]
        cand_texts = [v["text"] for k, v in pool_items]
        cand_metas = [v["meta"] for k, v in pool_items]
        cand_ids = [k for k, v in pool_items]

        # compute candidate embeddings (batch)
        if cand_texts:
            cand_embs = embedder.encode(cand_texts, convert_to_numpy=True)
            sel_idx_positions = mmr_select(q_emb, cand_embs, list(range(len(cand_texts))), k=MMR_K, lambda_param=0.7)
            sel_texts = [cand_texts[i] for i in sel_idx_positions]
            sel_metas = [cand_metas[i] for i in sel_idx_positions]
        else:
            sel_texts = []
            sel_metas = []

        # cross-encoder rerank if available (use for high precision)
        reranked = []
        if HAS_CE and sel_texts:
            try:
                reranker = CrossEncoder(RERANKER_MODEL)
                pairs = [[query, t] for t in sel_texts[:MMR_K]]
                scores = reranker.predict(pairs).tolist()
                reranked_pack = sorted(list(zip(sel_texts[:MMR_K], sel_metas[:MMR_K], scores)), key=lambda x: x[2], reverse=True)
                reranked = reranked_pack
            except Exception as e:
                logging.warning("Cross-encoder failed: %s", e)
        else:
            # fallback: use dot-product with query
            if sel_texts:
                sims = (np.array([np.dot(t_emb, q_emb) for t_emb in cand_embs[sel_idx_positions]]))
                reranked = sorted(list(zip(sel_texts, sel_metas, sims.tolist())), key=lambda x: x[2], reverse=True)

        # Prepare final context list (top FINAL_K)
        final_chunks = [r[0] for r in reranked[:FINAL_K]] if reranked else []
        final_metas = [r[1] for r in reranked[:FINAL_K]] if reranked else []
        rerank_scores = [r[2] for r in reranked[:FINAL_K]] if reranked else []

        # Decide: are we confident to answer from docs?
        confident = False
        if rerank_scores:
            # adaptive confidence decision
            s_sorted = sorted(rerank_scores, reverse=True)
            top = s_sorted[0]
            cond1 = top >= MIN_TOP_SCORE
            cond2 = True
            if len(s_sorted) >= 2:
                cond2 = (top - s_sorted[1]) >= DELTA_TOP2
            mu = float(np.mean(rerank_scores))
            sd = float(np.std(rerank_scores) if np.std(rerank_scores) > 1e-6 else 1.0)
            cond3 = (top - mu) >= MEAN_GAP_FACTOR * sd
            confident = cond1 and cond2 and cond3
            # but allow keyword-preserve: if required aliases present in final chunks, accept
            if not confident:
                if required_aliases_present(query, final_chunks):
                    logging.info("⚡ Keyword-preserve: required alias found in final chunks -> accept despite lower score")
                    confident = True

        # If not confident but there are candidate chunks that directly contain query keywords (exact match), accept them
        if not confident and final_chunks:
            q_keywords = extract_keywords(query)
            if any(any(kw in (txt or "").lower() for txt in final_chunks) for kw in q_keywords):
                logging.info("⚡ Exact keyword match present in retrieved chunks -> accept")
                confident = True

        # Build prompt
        if intent == "AI" and confident and final_chunks:
            # assemble context
            context = "\n\n---\n\n".join(final_chunks)
            # format instruction: comparisons should produce tables
            sys_instr = "You are a precise research assistant. Use ONLY the provided CONTEXT to answer. If the context lacks the answer, say \"Not found in documents\" and then give a concise general summary."
            format_hint = ""
            q_low = query.lower()
            if any(token in q_low for token in (" vs ", " difference", "compare", "vs.")):
                format_hint = "Format the main answer as a concise Markdown table comparing the items, then a 1-2 sentence summary."
            elif any(token in q_low for token in ("advantages", "disadvantages", "pros", "cons")):
                format_hint = "First give pros (bullets), then cons (bullets)."
            elif any(token in q_low for token in ("how to", "steps", "process")):
                format_hint = "Provide a numbered step-by-step procedure."

            prompt = f"{sys_instr}\n{format_hint}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
            answer = call_local_llm(prompt)
            source_str = ", ".join(
                sorted({
                    m.get("source", "unknown") + (f":p{m.get('page')}" if m.get("page") else "")
                    for m in final_metas
                })
            )

            print("\n💡 Answer (from docs):\n", answer.strip())
            print("\n📂 Source:", source_str or "unknown")
        else:
            # fallback general: answer with LLM without giving docs context
            logging.info("⚡ Fallback: answering with general LLM (not confident in docs)")
            # For CODE intent we can hint code blocks
            if intent == "CODING":
                prompt = f"You are a coding assistant. Provide a clean code example and brief explanation. Question: {query}"
            else:
                prompt = f"Answer concisely and helpfully: {query}"
            answer = call_local_llm(prompt)
            print("\n💡 Answer:\n", answer.strip())
            print("\n📂 Source: none (general)")

        logging.info("⏱️ Took %.2fs", now_s() - t0)
        print(f"\n⏱️ Took {now_s()-t0:.2f}s\n")

if __name__ == "__main__":
    main()














