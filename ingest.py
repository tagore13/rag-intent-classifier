import os
import argparse
import logging
import fitz  # PyMuPDF
import re
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ----------------------------
# Text cleaning
# ----------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    # Remove page numbers, headers, footers, extra spaces
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", " ", text)
    text = text.strip()
    return text

# ----------------------------
# Chunking with overlap
# ----------------------------
def chunk_text(text: str, max_len: int = 500, overlap: int = 100):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + max_len, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += max_len - overlap
    return chunks

# ----------------------------
# Extract text from PDF
# ----------------------------
def extract_pdf_chunks(pdf_path: str):
    doc = fitz.open(pdf_path)
    all_chunks = []
    for page_num, page in enumerate(doc, start=1):
        text = clean_text(page.get_text("text"))
        if not text:
            continue
        # Adaptive chunking
        chunks = chunk_text(text)
        for i, ch in enumerate(chunks):
            all_chunks.append({
                "text": ch,
                "metadata": {
                    "source": os.path.basename(pdf_path),
                    "page": page_num,
                    "chunk": i
                }
            })
    logging.info(f" -> extracted {len(all_chunks)} chunks")
    return all_chunks

# ----------------------------
# Ingest all PDFs
# ----------------------------
def ingest_all(docs_dir="docs", chroma_path="chroma_db", rebuild=False):
    logging.info("Use pytorch device_name: cpu")
    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    client = chromadb.PersistentClient(path=chroma_path)

    if rebuild:
        try:
            client.delete_collection("docs")
            logging.info("Existing collection dropped (rebuild mode).")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        "docs",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-mpnet-base-v2"),
    )

    all_chunks = []
    for fname in os.listdir(docs_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(docs_dir, fname)
        logging.info(f"Processing PDF: {path}")
        try:
            chunks = extract_pdf_chunks(path)
            all_chunks.extend(chunks)
        except Exception as e:
            logging.error(f"Failed to process {fname}: {e}")

    # Clean + filter texts before embedding
    texts = [str(c["text"]).strip() for c in all_chunks if c.get("text") and str(c["text"]).strip()]
    metadatas = [c["metadata"] for c in all_chunks if c.get("text") and str(c["text"]).strip()]
    ids = [f"{m['source']}_p{m['page']}_c{m['chunk']}" for m in metadatas]

    if not texts:
        logging.error("No valid text chunks found after cleaning. Aborting.")
        return

    logging.info(f"Computing embeddings for {len(texts)} chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Add to Chroma
    collection.add(documents=texts, embeddings=embeddings.tolist(), metadatas=metadatas, ids=ids)
    logging.info(f" Ingestion complete {len(texts)}/{len(texts)} chunks stored in collection 'docs'.")

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="docs", help="Directory with PDFs")
    parser.add_argument("--chroma", default="chroma_db", help="ChromaDB path")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild from scratch")
    args = parser.parse_args()

    ingest_all(docs_dir=args.dir, chroma_path=args.chroma, rebuild=args.rebuild)








