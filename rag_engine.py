"""
rag_engine.py — Fixed version
Problem: top-level imports of sentence_transformers + faiss crash the whole module
Fix: lazy imports inside functions so module always loads successfully
"""

def _load_deps():
    """Load heavy deps lazily — only when actually called."""
    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        import numpy as np
        return SentenceTransformer("all-MiniLM-L6-v2"), faiss, np
    except ImportError as e:
        raise ImportError(
            f"RAG engine requires: pip install sentence-transformers faiss-cpu\n{e}"
        )


def build_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("text_chunks cannot be empty")

    model, faiss, np = _load_deps()

    embeddings = model.encode(text_chunks)
    embeddings = np.array(embeddings, dtype=np.float32)

    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings


def retrieve(query, text_chunks, index, k=5):
    model, faiss, np = _load_deps()

    q_embedding = model.encode([query])
    q_embedding = np.array(q_embedding, dtype=np.float32)

    if q_embedding.ndim == 1:
        q_embedding = q_embedding.reshape(1, -1)

    q_embedding = np.ascontiguousarray(q_embedding, dtype=np.float32)

    k = min(k, len(text_chunks))
    _, indices = index.search(q_embedding, k)
    return [text_chunks[i] for i in indices[0] if i < len(text_chunks)]


# TEST
if __name__ == "__main__":
    texts = ["Hello world", "FAISS vector search", "Python programming"]
    try:
        index, emb = build_vector_store(texts)
        results = retrieve("coding", texts, index, k=2)
        print(f"Results: {results}")
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")