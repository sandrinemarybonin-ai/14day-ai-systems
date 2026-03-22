import os
import json
import glob
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from dotenv import load_dotenv
from mistralai import Mistral

DOCS_DIR = "data/docs"
INDEX_PATH = "data/index.json"
EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-small"  # adjust if needed


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(path: str) -> str:
    # Optional dependency: pypdf
    from pypdf import PdfReader
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_documents(docs_dir: str) -> List[Tuple[str, str]]:
    """
    Returns list of (doc_id, full_text)
    """
    docs: List[Tuple[str, str]] = []

    for path in glob.glob(os.path.join(docs_dir, "**/*"), recursive=True):
        if os.path.isdir(path):
            continue
        doc_id = os.path.relpath(path, docs_dir)

        if path.lower().endswith(".txt"):
            text = read_text_file(path)
            docs.append((doc_id, text))
        elif path.lower().endswith(".pdf"):
            try:
                text = read_pdf_file(path)
                docs.append((doc_id, text))
            except Exception as e:
                print(f"Skipping PDF (missing dependency or parse error): {doc_id} -> {e}")
        else:
            # Skip unknown formats in the first version
            continue

    return docs


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """
    Simple chunker by characters.
    Good enough for a first RAG. Improve later with token-based chunking.
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap  # overlap for continuity
        if start < 0:
            start = 0
    return chunks


def embed_texts(client: Mistral, texts: List[str]) -> np.ndarray:
    """
    Returns embeddings as a 2D numpy array: (n_texts, dim)
    """
    # Batch in chunks to avoid very large requests
    vectors = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, inputs=batch)
        batch_vecs = [d.embedding for d in resp.data]
        vectors.extend(batch_vecs)

    arr = np.array(vectors, dtype=np.float32)
    # Normalize for cosine similarity
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    arr = arr / norms
    return arr


def build_index(client: Mistral) -> Dict[str, Any]:
    docs = load_documents(DOCS_DIR)
    if not docs:
        raise RuntimeError(f"No documents found in {DOCS_DIR}. Add .txt files first.")

    chunks: List[Chunk] = []
    for doc_id, full_text in docs:
        for idx, chunk in enumerate(chunk_text(full_text)):
            chunks.append(Chunk(doc_id=doc_id, chunk_id=idx, text=chunk))

    texts = [c.text for c in chunks]
    embeddings = embed_texts(client, texts)

    index = {
        "embedding_model": EMBEDDING_MODEL,
        "chunks": [
            {"doc_id": c.doc_id, "chunk_id": c.chunk_id, "text": c.text}
            for c in chunks
        ],
        "embeddings": embeddings.tolist(),  # store as JSON-friendly list
    }
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f)
    return index


def load_index() -> Dict[str, Any]:
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError("Index not found. Run with --build-index first.")
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_top_k(client: Mistral, index: Dict[str, Any], query: str, k: int = 5) -> List[Dict[str, Any]]:
    q_vec = embed_texts(client, [query])[0]  # normalized
    embeddings = np.array(index["embeddings"], dtype=np.float32)

    # cosine similarity since vectors are normalized
    scores = embeddings @ q_vec
    top_idx = np.argsort(-scores)[:k]

    results = []
    for i in top_idx:
        chunk = index["chunks"][int(i)]
        results.append({
            "score": float(scores[int(i)]),
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
        })
    return results


def answer_with_context(client: Mistral, question: str, retrieved: List[Dict[str, Any]]) -> str:
    context_blocks = []
    for r in retrieved:
        context_blocks.append(
            f"[Source: {r['doc_id']} | chunk {r['chunk_id']}]\n{r['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "You are a document-grounded assistant.\n"
        "Answer using ONLY the provided context.\n"
        "If the answer is not in the context, say: 'I don't have enough information in the documents to answer that.'\n"
        "Cite sources by doc_id and chunk_id when you make a claim.\n"
    )

    user = f"QUESTION:\n{question}\n\nCONTEXT:\n{context}"

    resp = client.chat.complete(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return resp.choices[0].message.content.strip()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true", help="Build embeddings index from documents")
    parser.add_argument("--k", type=int, default=5, help="Top-k chunks to retrieve")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise SystemExit("Missing MISTRAL_API_KEY in .env")

    client = Mistral(api_key=api_key)

    if args.build_index:
        print("Building index...")
        build_index(client)
        print(f"Index saved to {INDEX_PATH}")
        return

    index = load_index()

    print("Document-Aware Assistant (type 'exit' to quit)")
    print("Tip: run once with --build-index after you add or change documents.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not q:
            print("Please enter a question.\n")
            continue

        retrieved = retrieve_top_k(client, index, q, k=args.k)
        print("\nTop retrieved chunks:")
        for r in retrieved:
            print(f"- score={r['score']:.3f} source={r['doc_id']} chunk={r['chunk_id']}")
        print()

        answer = answer_with_context(client, q, retrieved)
        print("Assistant:\n" + answer + "\n")


if __name__ == "__main__":
    main()