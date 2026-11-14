# ====================================================================
# main.py
# Flask web server: connects the frontend (static/index.html) to the
# vector database engine (vector_db.py), the MiniLM embedder and Groq.
#
# Endpoints and JSON field names are identical to the C++ original,
# so the existing index.html works without changes.
# ====================================================================

# ====================================================================
# PART 1 — IMPORTS AND ENVIRONMENT
# ====================================================================
import json
import os
import re
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from groq import Groq
from sentence_transformers import SentenceTransformer

from vector_db import (
    DIMS,
    DocumentDB,
    VectorDB,
    chunk_text,
    get_dist_fn,
    load_demo,
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Models
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"      # 384 dimensions
GEN_MODEL_NAME = "openai/gpt-oss-120b"

# Server
HOST = "0.0.0.0"
PORT = 8080
STATIC_DIR = "static"

# Defaults (same as the C++ version)
DEFAULT_K = 5                # demo search
DEFAULT_DOC_K = 3            # RAG search
PREVIEW_CHARS = 120

# Saved data (folder is created automatically; add data/ to .gitignore)
DATA_DIR = "data"
DOC_FILE = f"{DATA_DIR}/documents.json"        # document chunks + embeddings
MAP_FILE = f"{DATA_DIR}/map_points.json"       # dots shown on the map for documents


# ====================================================================
# PART 2 — CLIENTS AND DATABASES
# ====================================================================
print("Loading embedding model (first run downloads it)...")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
embed_lock = threading.Lock()
print(f"Embedding model ready: {EMBED_MODEL_NAME}")

groq_client = Groq(api_key=GROQ_API_KEY)

db = VectorDB(DIMS)
doc_db = DocumentDB(DOC_FILE, EMBED_MODEL_NAME)   # reloads saved documents
load_demo(db)

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app)


# ====================================================================
# PART 3 — HELPERS
# ====================================================================
def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Returns [] if embedding fails."""
    try:
        with embed_lock:
            vectors = embed_model.encode(texts)
        return [v.tolist() for v in vectors]
    except Exception as e:
        print(f"Embedding error: {e}")
        return []


def ask_llm(prompt: str) -> str:
    """Send a prompt to Groq. Raises on failure (caller handles it)."""
    response = groq_client.chat.completions.create(
        model=GEN_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def parse_vec(s: str) -> list[float]:
    """Parse '0.1,0.2,0.3' into floats, skipping invalid parts."""
    out: list[float] = []
    for part in (s or "").split(","):
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def round_vec(v) -> list[float]:
    """Round an embedding to 4 decimals for JSON output."""
    return [round(float(x), 4) for x in v]


def int_arg(value, default: int) -> int:
    """Convert to int, falling back to a default."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def body_json() -> dict:
    """Return the JSON body as a dict (empty dict if missing/invalid)."""
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def save_map_points() -> None:
    """Save the document dots of the map (category 'doc') to disk."""
    points = [
        {"metadata": v.metadata, "category": v.category,
         "embedding": [round(float(x), 4) for x in v.emb]}
        for v in db.all() if v.category == "doc"
    ]
    Path(MAP_FILE).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(MAP_FILE).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(points, f)
    os.replace(tmp, MAP_FILE)


def load_map_points() -> None:
    """Restore the document dots, but only for documents that still exist."""
    path = Path(MAP_FILE)
    if not path.exists():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            points = json.load(f)
        alive = {base_title(d.title) for d in doc_db.all()}
        restored = 0
        for p in points:
            if p["metadata"] in alive:
                db.insert(p["metadata"], p["category"], p["embedding"],
                          get_dist_fn("cosine"))
                restored += 1
        print(f"Restored {restored} map point(s)")
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"WARNING: could not read {MAP_FILE} ({e}). Starting without map points.")


def base_title(chunk_title: str) -> str:
    """'My Notes [2/3]' -> 'My Notes'."""
    return re.sub(r" \[\d+/\d+\]$", "", chunk_title)


def drop_map_point_if_orphaned(base: str) -> None:
    """Remove a document's map dot once none of its chunks remain."""
    if any(base_title(d.title) == base for d in doc_db.all()):
        return
    for v in db.all():
        if v.category == "doc" and v.metadata == base:
            db.remove(v.id)
    save_map_points()


def error(message: str):
    """JSON error response (HTTP 200, like the C++ version)."""
    return jsonify({"error": message})


load_map_points()          # restore document dots on the map


# ====================================================================
# PART 4 — DEMO VECTOR ENDPOINTS
# ====================================================================
@app.get("/search")
def search():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return error(f"need {DIMS}D vector")

    k = int_arg(request.args.get("k"), DEFAULT_K)
    metric = request.args.get("metric") or "cosine"
    algo = request.args.get("algo") or "hnsw"

    out = db.search(q, k, metric, algo)
    return jsonify({
        "results": [
            {
                "id": h.id,
                "metadata": h.meta,
                "category": h.cat,
                "distance": round(h.dist, 6),
                "embedding": round_vec(h.emb),
            }
            for h in out.hits
        ],
        "latencyUs": out.us,
        "algo": out.algo,
        "metric": out.metric,
    })


@app.post("/insert")
def insert():
    data = body_json()
    meta = data.get("metadata") or ""
    cat = data.get("category") or ""
    emb = data.get("embedding") or []
    if not meta or not isinstance(emb, list) or len(emb) != DIMS:
        return error("invalid body")
    try:
        emb = [float(x) for x in emb]
    except (TypeError, ValueError):
        return error("invalid body")

    new_id = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    if cat == "doc":
        save_map_points()
    return jsonify({"id": new_id})


@app.delete("/delete/<int:item_id>")
def delete(item_id: int):
    return jsonify({"ok": db.remove(item_id)})


@app.get("/items")
def items():
    return jsonify([
        {
            "id": v.id,
            "metadata": v.metadata,
            "category": v.category,
            "embedding": round_vec(v.emb),
        }
        for v in db.all()
    ])


@app.get("/benchmark")
def benchmark():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return error(f"need {DIMS}D vector")

    k = int_arg(request.args.get("k"), DEFAULT_K)
    metric = request.args.get("metric") or "cosine"

    b = db.benchmark(q, k, metric)
    return jsonify({
        "bruteforceUs": b.bf_us,
        "kdtreeUs": b.kd_us,
        "hnswUs": b.hnsw_us,
        "itemCount": b.n,
    })


@app.get("/hnsw-info")
def hnsw_info():
    gi = db.hnsw_info()
    return jsonify({
        "topLayer": gi.top_layer,
        "nodeCount": gi.node_count,
        "nodesPerLayer": gi.nodes_per_layer,
        "edgesPerLayer": gi.edges_per_layer,
        "nodes": [
            {"id": n.id, "metadata": n.metadata,
             "category": n.category, "maxLyr": n.max_lyr}
            for n in gi.nodes
        ],
        "edges": [
            {"src": e.src, "dst": e.dst, "lyr": e.lyr}
            for e in gi.edges
        ],
    })


# ====================================================================
# PART 5 — DOCUMENT + RAG ENDPOINTS
# ====================================================================
@app.post("/doc/insert")
def doc_insert():
    """Chunk the text, embed every chunk, store them in DocumentDB."""
    data = body_json()
    title = (data.get("title") or "").strip()
    text = data.get("text") or ""
    if not title or not text.strip():
        return error("need title and text")

    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)
    if len(embeddings) != len(chunks):
        return error("Embedding failed. Check that sentence-transformers "
                     "is installed and the model is available.")

    ids = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings), start=1):
        chunk_title = f"{title} [{i}/{len(chunks)}]" if len(chunks) > 1 else title
        ids.append(doc_db.insert(chunk_title, chunk, emb, persist=False))
    doc_db.save()                      # write once, after all chunks are in

    print(f"Inserted '{title}': {len(chunks)} chunk(s)")
    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})


@app.delete("/doc/delete/<int:doc_id>")
def doc_delete(doc_id: int):
    target = next((d for d in doc_db.all() if d.id == doc_id), None)
    ok = doc_db.remove(doc_id)
    if ok and target is not None:
        drop_map_point_if_orphaned(base_title(target.title))
    return jsonify({"ok": ok})


@app.get("/doc/list")
def doc_list():
    out = []
    for d in doc_db.all():
        preview = d.text[:PREVIEW_CHARS]
        if len(d.text) > PREVIEW_CHARS:
            preview += "…"
        out.append({
            "id": d.id,
            "title": d.title,
            "preview": preview,
            "words": d.text.count(" ") + 1,
        })
    return jsonify(out)


@app.post("/doc/search")
def doc_search():
    """Retrieval only (used by the map visualizer)."""
    data = body_json()
    question = (data.get("question") or "").strip()
    k = int_arg(data.get("k"), DEFAULT_DOC_K)
    if not question:
        return error("need question")

    q_emb = embed_texts([question])
    if not q_emb:
        return error("Embedding failed")

    hits = doc_db.search(q_emb[0], k)
    return jsonify({
        "contexts": [
            {"id": d.id, "title": d.title, "distance": round(dist, 4)}
            for dist, d in hits
        ]
    })


@app.post("/doc/ask")
def doc_ask():
    """Full RAG pipeline: embed -> retrieve -> build prompt -> generate."""
    data = body_json()
    question = (data.get("question") or "").strip()
    k = int_arg(data.get("k"), DEFAULT_DOC_K)
    if not question:
        return error("need question")

    # Step 1: embed the question
    q_emb = embed_texts([question])
    if not q_emb:
        return error("Embedding failed")

    # Step 2: retrieve top-k chunks
    hits = doc_db.search(q_emb[0], k)

    # Step 3: build the prompt (same wording as the original)
    context = "".join(
        f"[{i}] {d.title}:\n{d.text}\n\n"
        for i, (_, d) in enumerate(hits, start=1)
    )
    prompt = f"""You are a helpful assistant. Answer the user's question directly. Use the provided context if it contains relevant information. If it doesn't, just use your own general knowledge. IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. Just answer the question naturally.

Context:
{context}Question: {question}

Answer:"""

    # Step 4: generate the answer
    try:
        answer = ask_llm(prompt)
    except Exception as e:
        print(f"Groq error: {e}")
        return error(f"Groq request failed: {e}")

    # Step 5: return everything
    return jsonify({
        "answer": answer,
        "model": GEN_MODEL_NAME,
        "contexts": [
            {"id": d.id, "title": d.title, "text": d.text,
             "distance": round(dist, 4)}
            for dist, d in hits
        ],
        "docCount": doc_db.size(),
    })


# ====================================================================
# PART 6 — STATUS AND FRONTEND
# ====================================================================
@app.get("/status")
def status():
    # Field names kept from the Ollama version so index.html still works.
    # "ollamaAvailable" now means: Groq key loaded + embedding model ready.
    return jsonify({
        "ollamaAvailable": True,
        "embedModel": EMBED_MODEL_NAME,
        "genModel": GEN_MODEL_NAME,
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size(),
    })


@app.get("/stats")
def stats():
    return jsonify({
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
    })


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


# ====================================================================
# PART 7 — RUN
# ====================================================================
if __name__ == "__main__":
    print("=== VectorDB Engine ===")
    print(f"http://localhost:{PORT}")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Embed model: {EMBED_MODEL_NAME} | Gen model: {GEN_MODEL_NAME}")
    # debug=False: the reloader would load the embedding model twice
    app.run(host=HOST, port=PORT, debug=False)














