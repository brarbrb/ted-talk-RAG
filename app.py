# app.py
import os, json, time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import numpy as np
import requests
from pinecone import Pinecone

# ---- Load env ----
load_dotenv()

# ---- Config ----
BASE_URL = "https://api.llmod.ai/v1"
EMBED_MODEL = "RPRTHPB-text-embedding-3-small"
CHAT_MODEL = "RPRTHPB-gpt-5-mini"
EMBED_DIMS = 1536

# RAG hyperparameters (for reporting)
RAG_CHUNK_SIZE = 758         
RAG_OVERLAP_RATIO = 0.1
RAG_TOP_K = 20                


# ---- Keys ----
LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
HEADERS = {"Authorization": f"Bearer {LLMOD_API_KEY}", "Content-Type": "application/json"}

# ---- Pinecone ----

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "ted"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ---- Required system prompt (assignment) ----
REQUIRED_SYSTEM_PROMPT = """You are a TED Talk assistant that answers questions strictly and
only based on the TED dataset context provided to you (metadata
and transcript passages). You must not use any external
knowledge, the open internet, or information that is not explicitly
contained in the retrieved context. If the answer cannot be
determined from the provided context, respond: “I don't know
based on the provided TED data.” Always explain your answer
using the given context, quoting or paraphrasing the relevant
transcript or metadata when helpful.
"""

# =======================
# Embedding + retrieval
# =======================

def embed_texts_batch(texts, model=EMBED_MODEL, dims=EMBED_DIMS, max_retries=6):
    url = f"{BASE_URL}/embeddings"
    payload = {"model": model, "input": texts, "dimensions": dims}

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=60)
            if r.status_code == 200:
                data = r.json()
                embs = [np.array(item["embedding"], dtype=np.float32) for item in data["data"]]
                return np.vstack(embs)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 30))
                continue
            raise RuntimeError(f"Embeddings error {r.status_code}: {r.text[:500]}")
        except requests.RequestException as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Embeddings failed after retries. Last error: {last_err}")

def embed_query(text: str) -> np.ndarray:
    emb = embed_texts_batch([text], model=EMBED_MODEL, dims=EMBED_DIMS)
    if emb.shape != (1, EMBED_DIMS):
        raise ValueError(f"Bad query embedding shape: {emb.shape}")
    return emb[0].astype(np.float32)

def retrieve_from_pinecone(query: str, top_k: int):
    qvec = embed_query(query)
    res = index.query(
        vector=qvec.tolist(),
        top_k=top_k,
        include_metadata=True
    )

    matches = res.get("matches", res["matches"]) if isinstance(res, dict) else res.matches
    out = []
    for m in matches:
        mid = m.get("id") if isinstance(m, dict) else m.id
        score = m.get("score") if isinstance(m, dict) else m.score
        meta = m.get("metadata", {}) if isinstance(m, dict) else (m.metadata or {})
        out.append({
            "id": mid,
            "score": float(score) if score is not None else None,
            "metadata": meta
        })
    return out

def build_context(matches, max_chars: int = 12000) -> str:
    parts, total = [], 0
    for rank, item in enumerate(matches, start=1):
        md = item["metadata"] or {}
        title = str(md.get("title", ""))
        talk_id = str(md.get("talk_id", ""))
        chunk_id = md.get("chunk_id", "")
        text = str(md.get("text", ""))

        block = (
            f"[{rank}] talk_id={talk_id} | title={title} | chunk_id={chunk_id} | score={item['score']:.4f}\n"
            f"PASSAGE:\n{text}\n"
        )

        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n---\n".join(parts).strip()

# =======================
# Chat call
# =======================

def call_chat_model(question: str, context: str):
    """
    Returns: (answer_text, augmented_prompt_dict)
    IMPORTANT: Do NOT pass temperature for gpt-5 models (or keep it at 1 only).
    """
    url = f"{BASE_URL}/chat/completions"

    system_msg = REQUIRED_SYSTEM_PROMPT
    user_msg = f"TED DATA CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
    }

    max_retries = 6
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, headers=HEADERS, data=json.dumps(payload), timeout=90)
            if r.status_code == 200:
                data = r.json()
                answer = data["choices"][0]["message"]["content"]
                return answer, {"System": system_msg, "User": user_msg}
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(2 ** attempt, 30))
                continue
            raise RuntimeError(f"Chat error {r.status_code}: {r.text[:800]}")
        except requests.RequestException as e:
            last_err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"Chat failed after retries. Last error: {last_err}")

# =======================
# RAG end-to-end
# =======================

def rag_answer(question: str, top_k: int):
    matches = retrieve_from_pinecone(question, top_k=top_k)
    # context array in the exact output format the assignment wants
    context_array = []
    for m in matches:
        md = m["metadata"] or {}
        context_array.append({
            "talk_id": str(md.get("talk_id", "")),
            "title": str(md.get("title", "")),
            "chunk": str(md.get("text", "")),
            "score": float(m["score"]) if m["score"] is not None else None
        })

    context_text = build_context(matches)
    answer, augmented_prompt = call_chat_model(question, context_text)

    return answer, context_array, augmented_prompt

# =======================
# Flask API
# =======================

app = Flask(__name__)

@app.get("/api/stats")
def api_stats():
    return jsonify({
        "chunk_size": int(RAG_CHUNK_SIZE),
        "overlap_ratio": float(RAG_OVERLAP_RATIO),
        "top_k": int(RAG_TOP_K),
    })

@app.post("/api/prompt")
def api_prompt():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Missing 'question' in request JSON."}), 400

    try:
        response_text, context_array, augmented_prompt = rag_answer(question, top_k=RAG_TOP_K)
        return jsonify({
            "response": response_text,
            "context": context_array,
            "Augmented_prompt": augmented_prompt
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
