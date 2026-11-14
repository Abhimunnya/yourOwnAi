# ====================================================================
# vector_db.py
# From-scratch vector database engine (Python port of the C++ main.cpp)
#
# Contains:  distance metrics, BruteForce, KDTree, HNSW,
#            VectorDB (16D demo index), DocumentDB (real embeddings),
#            text chunker, demo data.
#
# No Flask / Groq / sentence-transformers code lives here, so this
# module can be tested and reused without the web server.
# ====================================================================

# ====================================================================
# PART 1 — IMPORTS AND CONSTANTS
# ====================================================================
import heapq
import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

DIMS = 16                      # dimension of the demo vectors

# HNSW parameters
HNSW_M = 16                    # max neighbours per node (layers > 0)
HNSW_EF_BUILD = 200            # candidate list size while inserting
HNSW_EF_SEARCH = 50            # candidate list size while searching
HNSW_SEED = 42                 # fixed seed -> reproducible level generation

# Document database parameters
DOC_BRUTE_FORCE_LIMIT = 10     # below this many chunks, use brute force
DOC_MAX_DIST = 0.7             # ignore chunks with cosine distance above this

# Chunker parameters
CHUNK_WORDS = 250              # words per chunk
CHUNK_OVERLAP_WORDS = 30       # words shared between neighbouring chunks

# Type aliases
Vector = np.ndarray
DistFn = Callable[[Vector, Vector], float]
Pair = tuple[float, int]       # (distance, item id)


# ====================================================================
# PART 2 — DATA TYPES
# ====================================================================
def to_vector(values) -> Vector:
    """Convert any list/array of numbers into a float32 numpy vector."""
    return np.asarray(values, dtype=np.float32)


@dataclass(eq=False)
class VectorItem:
    """One stored vector (equivalent of the C++ VectorItem struct)."""
    id: int
    metadata: str
    category: str
    emb: Vector


@dataclass(eq=False)
class DocItem:
    """One stored document chunk with its real embedding."""
    id: int
    title: str
    text: str
    emb: Vector


@dataclass
class Hit:
    """One search result returned by VectorDB.search()."""
    id: int
    meta: str
    cat: str
    emb: Vector
    dist: float


@dataclass
class SearchOut:
    """Search results plus timing information."""
    hits: list[Hit]
    us: int
    algo: str
    metric: str


@dataclass
class BenchOut:
    """Timing of all three algorithms for the same query (microseconds)."""
    bf_us: int
    kd_us: int
    hnsw_us: int
    n: int


@dataclass
class NodeView:
    id: int
    metadata: str
    category: str
    max_lyr: int


@dataclass
class EdgeView:
    src: int
    dst: int
    lyr: int


@dataclass
class GraphInfo:
    """Summary of the HNSW graph, used by the UI layer view."""
    top_layer: int
    node_count: int
    nodes_per_layer: list[int]
    edges_per_layer: list[int]
    nodes: list[NodeView] = field(default_factory=list)
    edges: list[EdgeView] = field(default_factory=list)


# ====================================================================
# PART 3 — DISTANCE METRICS
# ====================================================================
def euclidean(a: Vector, b: Vector) -> float:
    """Straight-line distance."""
    return float(np.linalg.norm(a - b))


def cosine(a: Vector, b: Vector) -> float:
    """Cosine distance = 1 - cosine similarity."""
    dot = float(np.dot(a, b))
    na = float(np.dot(a, a))
    nb = float(np.dot(b, b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a: Vector, b: Vector) -> float:
    """Sum of absolute differences."""
    return float(np.abs(a - b).sum())


def get_dist_fn(metric: str) -> DistFn:
    """Return the distance function for a metric name (default: euclidean)."""
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean


# ====================================================================
# PART 4 — BRUTE FORCE
# ====================================================================
class BruteForce:
    """Exact search: compute every distance, sort, take the first k."""

    def __init__(self) -> None:
        self.items: list[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: Vector, k: int, dist: DistFn) -> list[Pair]:
        results = sorted((dist(q, v.emb), v.id) for v in self.items)
        return results[:max(k, 0)]

    def remove(self, item_id: int) -> None:
        self.items = [v for v in self.items if v.id != item_id]


# ====================================================================
# PART 5 — KD-TREE
# ====================================================================
class KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem) -> None:
        self.item = item
        self.left: Optional["KDNode"] = None
        self.right: Optional["KDNode"] = None


class KDTree:
    """K-dimensional tree with branch pruning (same logic as the C++ code)."""

    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.root: Optional[KDNode] = None

    def insert(self, v: VectorItem) -> None:
        new_node = KDNode(v)
        if self.root is None:
            self.root = new_node
            return
        node, depth = self.root, 0
        while True:
            ax = depth % self.dims
            go_left = v.emb[ax] < node.item.emb[ax]
            child = node.left if go_left else node.right
            if child is None:
                if go_left:
                    node.left = new_node
                else:
                    node.right = new_node
                return
            node, depth = child, depth + 1

    def _knn(self, node: Optional[KDNode], q: Vector, k: int, depth: int,
             dist: DistFn, heap: list[tuple[float, int]]) -> None:
        """Recursive search. `heap` is a max-heap stored as (-dist, -id)."""
        if node is None:
            return
        dn = dist(q, node.item.emb)
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, -node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)

        ax = depth % self.dims
        diff = float(q[ax] - node.item.emb[ax])
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, q, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist, heap)

    def knn(self, q: Vector, k: int, dist: DistFn) -> list[Pair]:
        if k <= 0:
            return []
        heap: list[tuple[float, int]] = []
        self._knn(self.root, q, k, 0, dist, heap)
        return sorted((-d, -i) for d, i in heap)

    def rebuild(self, items: list[VectorItem]) -> None:
        self.root = None
        for v in items:
            self.insert(v)


# ====================================================================
# PART 6 — HNSW (Hierarchical Navigable Small World)
# ====================================================================
@dataclass(eq=False)
class HNSWNode:
    item: VectorItem
    max_lyr: int
    nbrs: list[list[int]]          # nbrs[layer] = neighbour ids


class HNSW:
    """Multi-layer navigable small-world graph for approximate kNN."""

    def __init__(self, m: int = HNSW_M, ef_build: int = HNSW_EF_BUILD,
                 seed: int = HNSW_SEED) -> None:
        self.m = m
        self.m0 = 2 * m
        self.ef_build = ef_build
        self.ml = 1.0 / math.log(m)
        self.top_layer = -1
        self.entry_pt = -1
        self.graph: dict[int, HNSWNode] = {}
        self._rng = random.Random(seed)

    # ---------- internal helpers ----------
    def _rand_level(self) -> int:
        u = 1.0 - self._rng.random()           # in (0, 1], avoids log(0)
        return int(math.floor(-math.log(u) * self.ml))

    def _search_layer(self, q: Vector, ep: int, ef: int, lyr: int,
                      dist: DistFn) -> list[Pair]:
        """Greedy best-first search inside one layer. Returns sorted pairs."""
        d0 = dist(q, self.graph[ep].item.emb)
        visited = {ep}
        cands: list[Pair] = [(d0, ep)]                    # min-heap
        found: list[tuple[float, int]] = [(-d0, -ep)]     # max-heap

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            layers = self.graph[cid].nbrs
            if lyr >= len(layers):
                continue
            for nid in layers[lyr]:
                if nid in visited or nid not in self.graph:
                    continue
                visited.add(nid)
                nd = dist(q, self.graph[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, -nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        return sorted((-d, -i) for d, i in found)

    @staticmethod
    def _select_nbrs(cands: list[Pair], max_m: int) -> list[int]:
        """Pick the closest `max_m` candidates (simple selection)."""
        return [i for _, i in cands[:max_m]]

    # ---------- public API ----------
    def insert(self, item: VectorItem, dist: DistFn) -> None:
        item_id = item.id
        lvl = self._rand_level()
        self.graph[item_id] = HNSWNode(item, lvl, [[] for _ in range(lvl + 1)])

        if self.entry_pt == -1:
            self.entry_pt = item_id
            self.top_layer = lvl
            return

        ep = self.entry_pt
        # Phase 1: greedy descent through layers above the new node's level
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.graph[ep].nbrs):
                w = self._search_layer(item.emb, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]

        # Phase 2: connect the node on every layer it belongs to
        for lc in range(min(self.top_layer, lvl), -1, -1):
            w = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.m0 if lc == 0 else self.m
            sel = self._select_nbrs(w, max_m)
            self.graph[item_id].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.graph:
                    continue
                n_nbrs = self.graph[nid].nbrs
                while len(n_nbrs) <= lc:
                    n_nbrs.append([])
                conn = n_nbrs[lc]
                conn.append(item_id)
                if len(conn) > max_m:           # prune to the closest max_m
                    n_emb = self.graph[nid].item.emb
                    ds = sorted(
                        (dist(n_emb, self.graph[c].item.emb), c)
                        for c in conn if c in self.graph
                    )
                    conn[:] = [c for _, c in ds[:max_m]]
            if w:
                ep = w[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = item_id

    def knn(self, q: Vector, k: int, ef: int, dist: DistFn) -> list[Pair]:
        if self.entry_pt == -1 or k <= 0:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.graph[ep].nbrs):
                w = self._search_layer(q, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]
        w = self._search_layer(q, ep, max(ef, k), 0, dist)
        return w[:k]

    def remove(self, item_id: int) -> None:
        """Delete a node and every link pointing to it (same as the C++ code)."""
        if item_id not in self.graph:
            return
        for node in self.graph.values():
            for layer in node.nbrs:
                layer[:] = [n for n in layer if n != item_id]
        if self.entry_pt == item_id:
            self.entry_pt = -1
            for nid in self.graph:
                if nid != item_id:
                    self.entry_pt = nid
                    break
        del self.graph[item_id]

    def get_info(self) -> GraphInfo:
        max_l = max(self.top_layer + 1, 1)
        info = GraphInfo(
            top_layer=self.top_layer,
            node_count=len(self.graph),
            nodes_per_layer=[0] * max_l,
            edges_per_layer=[0] * max_l,
        )
        for node_id, node in self.graph.items():
            info.nodes.append(NodeView(node_id, node.item.metadata,
                                       node.item.category, node.max_lyr))
            for lc in range(node.max_lyr + 1):
                if lc >= max_l:
                    break
                info.nodes_per_layer[lc] += 1
                if lc < len(node.nbrs):
                    for nid in node.nbrs[lc]:
                        if node_id < nid:
                            info.edges_per_layer[lc] += 1
                            info.edges.append(EdgeView(node_id, nid, lc))
        return info

    def size(self) -> int:
        return len(self.graph)


# ====================================================================
# PART 7 — VECTOR DATABASE (demo 16D index)
# ====================================================================
class VectorDB:
    """Demo database: keeps the same vectors in all three indexes."""

    def __init__(self, dims: int = DIMS) -> None:
        self.dims = dims
        self._store: dict[int, VectorItem] = {}
        self._bf = BruteForce()
        self._kdt = KDTree(dims)
        self._hnsw = HNSW(HNSW_M, HNSW_EF_BUILD)
        self._lock = threading.Lock()
        self._next_id = 1

    def insert(self, meta: str, cat: str, emb, dist: DistFn) -> int:
        with self._lock:
            item = VectorItem(self._next_id, meta, cat, to_vector(emb))
            self._next_id += 1
            self._store[item.id] = item
            self._bf.insert(item)
            self._kdt.insert(item)
            self._hnsw.insert(item, dist)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._bf.remove(item_id)
            self._hnsw.remove(item_id)
            self._kdt.rebuild(list(self._store.values()))
            return True

    def search(self, q, k: int, metric: str, algo: str) -> SearchOut:
        q = to_vector(q)
        with self._lock:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter_ns()
            if algo == "bruteforce":
                raw = self._bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self._kdt.knn(q, k, dfn)
            else:
                raw = self._hnsw.knn(q, k, HNSW_EF_SEARCH, dfn)
            us = (time.perf_counter_ns() - t0) // 1000

            hits = [
                Hit(i, self._store[i].metadata, self._store[i].category,
                    self._store[i].emb, d)
                for d, i in raw if i in self._store
            ]
            return SearchOut(hits, us, algo, metric)

    def benchmark(self, q, k: int, metric: str) -> BenchOut:
        q = to_vector(q)
        with self._lock:
            dfn = get_dist_fn(metric)

            def timed(fn: Callable[[], object]) -> int:
                t = time.perf_counter_ns()
                fn()
                return (time.perf_counter_ns() - t) // 1000

            return BenchOut(
                bf_us=timed(lambda: self._bf.knn(q, k, dfn)),
                kd_us=timed(lambda: self._kdt.knn(q, k, dfn)),
                hnsw_us=timed(lambda: self._hnsw.knn(q, k, HNSW_EF_SEARCH, dfn)),
                n=len(self._store),
            )

    def all(self) -> list[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self) -> GraphInfo:
        with self._lock:
            return self._hnsw.get_info()

    def size(self) -> int:
        with self._lock:
            return len(self._store)


# ====================================================================
# PART 8 — DOCUMENT DATABASE (HNSW over real embeddings)
# ====================================================================
class DocumentDB:
    """Stores document chunks with real embeddings; cosine distance.

    If `path` is given, chunks are saved to that JSON file after every
    change and reloaded on startup (the HNSW graph is rebuilt from them).
    """

    def __init__(self, path: Optional[str] = None, model_name: str = "") -> None:
        self._store: dict[int, DocItem] = {}
        self._hnsw = HNSW(HNSW_M, HNSW_EF_BUILD)
        self._bf = BruteForce()          # exact fallback for small sets
        self._lock = threading.Lock()
        self._next_id = 1
        self.dims = 0                    # set by the first inserted embedding
        self._path = Path(path) if path else None
        self._model_name = model_name
        self._load()

    # ---------- internal helpers ----------
    def _add(self, item: DocItem) -> None:
        """Put an item into the store and both indexes (no locking, no saving)."""
        self._store[item.id] = item
        vi = VectorItem(item.id, item.title, "doc", item.emb)
        self._hnsw.insert(vi, cosine)
        self._bf.insert(vi)

    def _write(self) -> None:
        """Save everything to the JSON file (caller must hold the lock)."""
        if self._path is None:
            return
        data = {
            "model": self._model_name,
            "dims": self.dims,
            "next_id": self._next_id,
            "items": [
                {"id": it.id, "title": it.title, "text": it.text,
                 "emb": [round(float(x), 6) for x in it.emb]}
                for it in self._store.values()
            ],
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self._path)      # atomic: never leaves a half-written file

    def _load(self) -> None:
        """Load saved chunks from disk and rebuild the indexes."""
        if self._path is None or not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            saved_model = data.get("model", "")
            if self._model_name and saved_model and saved_model != self._model_name:
                print(f"WARNING: {self._path} was made with '{saved_model}', "
                      f"current model is '{self._model_name}'. Ignoring saved documents.")
                return
            items = [DocItem(d["id"], d["title"], d["text"], to_vector(d["emb"]))
                     for d in sorted(data["items"], key=lambda d: d["id"])]
        except (OSError, ValueError, KeyError, TypeError) as e:
            bad = self._path.with_suffix(".bad")
            print(f"WARNING: could not read {self._path} ({e}). Moved to {bad}.")
            os.replace(self._path, bad)
            return

        for item in items:
            self._add(item)
        top_id = max((it.id for it in items), default=0)
        self._next_id = max(int(data.get("next_id", 1)), top_id + 1)
        self.dims = int(data.get("dims", 0)) or (len(items[0].emb) if items else 0)
        print(f"Loaded {len(items)} saved document chunk(s) from {self._path}")

    # ---------- public API ----------
    def insert(self, title: str, text: str, emb, persist: bool = True) -> int:
        """Store one chunk with its pre-computed embedding. Returns its id.

        Pass persist=False when inserting many chunks, then call save() once.
        """
        with self._lock:
            vec = to_vector(emb)
            if self.dims == 0:
                self.dims = len(vec)
            item = DocItem(self._next_id, title, text, vec)
            self._next_id += 1
            self._add(item)
            if persist:
                self._write()
            return item.id

    def save(self) -> None:
        """Write the current state to disk."""
        with self._lock:
            self._write()

    def search(self, q, k: int,
               max_dist: float = DOC_MAX_DIST) -> list[tuple[float, DocItem]]:
        """Return the top-k most similar chunks within `max_dist`."""
        q = to_vector(q)
        with self._lock:
            if not self._store:
                return []
            if len(self._store) < DOC_BRUTE_FORCE_LIMIT:
                raw = self._bf.knn(q, k, cosine)
            else:
                raw = self._hnsw.knn(q, k, HNSW_EF_SEARCH, cosine)
            return [(d, self._store[i]) for d, i in raw
                    if i in self._store and d <= max_dist]

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._hnsw.remove(item_id)
            self._bf.remove(item_id)
            self._write()
            return True

    def all(self) -> list[DocItem]:
        with self._lock:
            return list(self._store.values())

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def get_dims(self) -> int:
        return self.dims


# ====================================================================
# PART 9 — TEXT CHUNKER
# ====================================================================
def chunk_text(text: str, chunk_words: int = CHUNK_WORDS,
               overlap_words: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping word-based chunks."""
    step = chunk_words - overlap_words
    if step <= 0:
        raise ValueError("overlap_words must be smaller than chunk_words")

    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks: list[str] = []
    for start in range(0, len(words), step):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
    return chunks


# ====================================================================
# PART 10 — DEMO DATA (16D categorical vectors)
# ====================================================================
# Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
DEMO_DATA: list[tuple[str, str, list[float]]] = [
    ("Linked List: nodes connected by pointers", "cs",
     [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]),
    ("Binary Search Tree: O(log n) search and insert", "cs",
     [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10]),
    ("Dynamic Programming: memoization overlapping subproblems", "cs",
     [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07]),
    ("Graph BFS and DFS: breadth and depth first traversal", "cs",
     [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07]),
    ("Hash Table: O(1) lookup with collision chaining", "cs",
     [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09]),
    ("Calculus: derivatives integrals and limits", "math",
     [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10]),
    ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
     [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09]),
    ("Probability: distributions random variables Bayes theorem", "math",
     [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08]),
    ("Number Theory: primes modular arithmetic RSA cryptography", "math",
     [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06]),
    ("Combinatorics: permutations combinations generating functions", "math",
     [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07]),
    ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
     [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07]),
    ("Sushi: vinegared rice raw fish and nori rolls", "food",
     [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08]),
    ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
     [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06]),
    ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
     [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09]),
    ("Croissant: laminated pastry with buttery flaky layers", "food",
     [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06]),
    ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
     [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72]),
    ("Football: tackles touchdowns field goals and strategy", "sports",
     [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76]),
    ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
     [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82]),
    ("Chess: openings endgames tactics strategic board game", "sports",
     [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90]),
    ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
     [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80]),
]


def load_demo(db: VectorDB) -> None:
    """Insert the 20 demo vectors into a VectorDB."""
    dist = get_dist_fn("cosine")
    for meta, cat, emb in DEMO_DATA:
        db.insert(meta, cat, emb, dist)


# ====================================================================
# PART 11 — QUICK SELF-TEST  (run:  uv run python vector_db.py)
# ====================================================================
if __name__ == "__main__":
    demo_db = VectorDB(DIMS)
    load_demo(demo_db)
    print(f"Loaded {demo_db.size()} demo vectors")

    query = DEMO_DATA[1][2]          # same vector as "Binary Search Tree"
    for algorithm in ("bruteforce", "kdtree", "hnsw"):
        out = demo_db.search(query, 3, "euclidean", algorithm)
        top = out.hits[0]
        print(f"{algorithm:10s} -> {top.meta[:30]!r}  "
              f"dist={top.dist:.4f}  time={out.us} us")

    sample = " ".join(f"w{i}" for i in range(600))
    print(f"Chunker: 600 words -> {len(chunk_text(sample))} chunks")




























