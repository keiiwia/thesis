"""Local server: the same stream feeds two learners.

Memory: a kNN datastore over the frozen base model. Provenance is exact.
Weights: one LoRA adapter per user, trained on their words. Real learning, real forgetting,
         and provenance that can only be estimated by switching adapters off.

Run:  uv run python server.py          (Gemma-3 270M IT on CPU)
"""
import os, re, json, threading, queue, time, traceback
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import knn_core as kc

HERE = os.path.dirname(os.path.abspath(__file__))
LAM = float(os.environ.get("LAM", 0.25))
K = int(os.environ.get("K", 16))
MAX_NEW = int(os.environ.get("MAX_NEW", 60))
PORT = int(os.environ.get("PORT", 8000))
VIEWER_PORT = int(os.environ.get("VIEWER_PORT", 8046))
WEAK_MATCH = 3.0          # nearest neighbor farther than 3x the typical distance
MAX_CHARS = 1000
DATA = os.path.join(HERE, "data")
GRAPHS, PTS = os.path.join(DATA, "graph_files"), os.path.join(DATA, "graphs_pt")
for d in (DATA, GRAPHS, PTS):
    os.makedirs(d, exist_ok=True)
DS_PATH = os.path.join(DATA, "datastore")
ADAPTER_PATH = os.path.join(DATA, "adapters.pt")
LOG_PATH = os.path.join(DATA, "weights_log.json")

from backend_real import Backend

print("Loading model...", flush=True)
be = Backend()
lock = threading.Lock()      # one model call at a time; adapters must not change mid-answer
PROBES = [tuple(x.strip() for x in l.split("|", 1)) for l in open(os.path.join(HERE, "probes.txt")) if "|" in l]


def ingest_memory(user_id, text):
    keys, vals, spans = be.encode_ingest(text)
    if len(vals) == 0:
        raise HTTPException(400, "Text is too short to store.")
    return ds.add_entry(user_id, text, keys, vals, spans, extra={"created": time.time()})


# ---------- memory ----------
# Memory holds only what people write. Older datastores were seeded with background text; drop it.
ds = kc.Datastore.load(DS_PATH, be.name) or kc.Datastore(be.dim)
if not (len(ds.keys) == len(ds.values) == len(ds.meta)):
    # The two saved files disagree (an interrupted or overlapping save). The texts survive, so re-encode them.
    print(f"Datastore files disagree ({len(ds.keys)} keys, {len(ds.meta)} rows); re-encoding stored texts...", flush=True)
    old, ds = ds, kc.Datastore(be.dim)
    for e in old.entries:
        ds.next_id = e["entry_id"]
        ds.add_entry(e["user_id"], e["text"], *be.encode_ingest(e["text"]), extra=e)
    ds.next_id = old.next_id
for e in [e for e in ds.entries if e["user_id"] == "background"]:
    ds.remove_entry(e["entry_id"])
ds.save(DS_PATH, be.name)
TEMP = kc.calibrate_temp(ds)


def stream_entries():
    return sorted((e for e in ds.entries if e["user_id"] != "background"), key=lambda e: e["entry_id"])


def stream_users():
    return [u for u in ds.users() if u != "background"]


# ---------- weights ----------
def probe_scores(users):
    be.set_active(users)
    return [be.score(q, a) for q, a in PROBES]


BASE_PROBES = probe_scores([])
WLOG = {"version": 0, "history": [], "trained": {}}   # trained: entry_id -> training stats


def record(event):
    cur = probe_scores(be.adapter_users())
    drops = [b - c for b, c in zip(BASE_PROBES, cur)]
    WLOG["version"] += 1
    WLOG["history"].append({
        "t": time.time(), "event": event, "adapters": be.adapter_users(),
        "mean_drop": float(np.mean(drops)),
        "probes": [{"q": q, "a": a, "base": b, "now": c}
                   for (q, a), b, c in zip(PROBES, BASE_PROBES, cur)]})
    be.save_adapters(ADAPTER_PATH)
    json.dump(WLOG, open(LOG_PATH, "w"))


def restore_weights():
    global WLOG
    restored = be.load_adapters(ADAPTER_PATH)
    want = list(dict.fromkeys(e["user_id"] for e in stream_entries()))
    if restored is not None and restored == want and os.path.exists(LOG_PATH):
        WLOG = json.load(open(LOG_PATH))
        WLOG["trained"] = {int(k): v for k, v in WLOG["trained"].items()}
        print(f"Restored {len(restored)} adapters.", flush=True)
        return
    for u in be.adapter_users():
        be.remove_adapter(u)
    if want:
        print("Retraining adapters from stored entries, in the order they arrived...", flush=True)
    for e in stream_entries():
        WLOG["trained"][e["entry_id"]] = be.train(e["user_id"], [e["text"]])
        record(f"{e['user_id']} added text (replayed at startup)")
    if not WLOG["history"]:
        record("start")


restore_weights()
QUERIES, TRACES = {}, {}
trace_q = queue.Queue()
print(f"Datastore: {len(ds)} entries, temp {TEMP:.2f}. Adapters: {be.adapter_users()}", flush=True)


# ---------- generation ----------
def top_k(p, n=5):
    idx = np.argsort(-p)[:n]
    return [{"id": int(i), "text": be.decode([int(i)]), "p": float(p[i])} for i in idx]


def repeating(gen, n=6):
    """Greedy decoding on a small model loops. Stop once the last n tokens already appeared."""
    if len(gen) < 2 * n:
        return False
    tail = gen[-n:]
    return any(gen[i:i + n] == tail for i in range(len(gen) - 2 * n + 1))


def piece_of(gen, t):
    prev = be.decode(gen)
    return be.decode(gen + [t])[len(prev):]


def generate_memory(ids, lam, exclude=(), record_steps=False):
    be.set_active([])                      # memory always runs on the untouched base model
    ids, gen, steps = list(ids), [], []
    users = [u for u in stream_users() if u not in exclude]
    for _ in range(MAX_NEW):
        h, p_lm = be.step(ids)
        st = kc.knn_step(ds, h, p_lm, lam, K, TEMP, exclude)
        t = int(np.argmax(st["p_final"]))
        if t in be.stop_ids:
            break
        piece = piece_of(gen, t)
        if record_steps:
            shares, p_t = kc.decompose(ds, st, p_lm, t)
            lm_top = int(np.argmax(p_lm))
            neighbors = []
            for i, w, d2 in zip(st["idx"], st["w"], st["d2"]):
                m = ds.meta[i]
                text = next(e["text"] for e in ds.entries if e["entry_id"] == m["entry_id"])
                s, e = m["char_start"], m["char_end"]
                neighbors.append({
                    "user_id": m["user_id"], "entry_id": m["entry_id"],
                    "value_text": text[s:e], "context_before": text[max(0, s - 120):s],
                    "context_after": text[e:e + 60], "weight": float(w),
                    "d2_over_temp": float(d2 / TEMP), "supports_chosen": bool(ds.values[i] == t)})
            cf = {}
            for u in users:
                pc, top, ptop = kc.knn_point(ds, h, p_lm, lam, K, TEMP, set(exclude) | {u}, t, lm_top)
                cf[u] = {"p_chosen": pc, "top_text": be.decode([top]), "top_p": ptop, "flips": top != t}
            near = float(st["d2"][0] / TEMP) if len(st["d2"]) else None
            steps.append({
                "pos": len(gen), "token_id": t, "text": piece, "p_final": p_t,
                "p_lm": float(p_lm[t]), "shares": shares, "knn_share": 1.0 - shares["parametric"],
                "final_top5": top_k(st["p_final"]), "lm_top5": top_k(p_lm),
                "entropy_final": kc.entropy(st["p_final"]), "entropy_lm": kc.entropy(p_lm),
                "nearest_d2_over_temp": near, "weak_match": near is not None and near > WEAK_MATCH,
                "neighbors": neighbors, "counterfactual": cf,
                "_trace": {"prefix_ids": list(ids), "target_ids": sorted({t, lm_top})}})
        gen.append(t); ids.append(t)
        if repeating(gen):
            break
    return be.decode(gen), steps


def generate_weights(ids, active, record_steps=False):
    ids, gen, steps = list(ids), [], []
    for _ in range(MAX_NEW):
        be.set_active(active)
        _, p = be.step(ids)
        t = int(np.argmax(p))
        if t in be.stop_ids:
            break
        piece = piece_of(gen, t)
        if record_steps:
            be.set_active([])
            _, p0 = be.step(ids)
            steps.append({
                "pos": len(gen), "token_id": t, "text": piece,
                "p_tuned": float(p[t]), "p_base": float(p0[t]),
                # share of this word's probability the base model can't account for (0 if the base liked it more)
                "lift": max(0.0, 1.0 - float(p0[t]) / float(p[t])),
                "tuned_top5": top_k(p), "base_top5": top_k(p0), "_ids": list(ids)})
        gen.append(t); ids.append(t)
        if repeating(gen):
            break
    return be.decode(gen), steps


def strip_private(steps, keys):
    return [{k: v for k, v in s.items() if k not in keys} for s in steps]


def public_query(q):
    return {"query_id": q["query_id"], "user_id": q["user_id"], "question": q["question"],
            "created": q["created"],
            "memory": {**{k: v for k, v in q["memory"].items() if k != "steps"},
                       "steps": strip_private(q["memory"]["steps"], {"_trace"})},
            "weights": {**{k: v for k, v in q["weights"].items() if k != "steps"},
                        "steps": strip_private(q["weights"]["steps"], {"_ids"})}}


# ---------- attribution graphs ----------
def trace_worker():
    while True:
        slug = trace_q.get()
        job = TRACES[slug]
        job["status"] = "running"; job["started"] = time.time()
        try:
            be.trace(job["prefix_ids"], job["target_ids"], slug, GRAPHS, PTS)
            job["status"] = "done"
        except Exception as ex:
            job["status"] = "error"; job["error"] = f"{type(ex).__name__}: {ex}"
            traceback.print_exc()
        job["finished"] = time.time()


threading.Thread(target=trace_worker, daemon=True).start()
VIEWER = None
if be.supports_trace:
    try:
        from circuit_tracer.frontend.local_server import serve
        serve(data_dir=GRAPHS, port=VIEWER_PORT)
        VIEWER = f"http://localhost:{VIEWER_PORT}/index.html"
    except Exception as ex:
        print("Graph viewer not started:", ex, flush=True)


# ---------- API ----------
app = FastAPI()
NAME_RE = re.compile(r"^[A-Za-z0-9_ .-]{1,24}$")


class IngestIn(BaseModel):
    user_id: str
    text: str
    consent: bool


class AskIn(BaseModel):
    user_id: str
    question: str


class RegenIn(BaseModel):
    query_id: int
    mode: str = "memory"            # "memory" or "weights"
    exclude_users: list[str] = []
    base_only: bool = False


class TokenIn(BaseModel):
    query_id: int
    pos: int


def check_name(u):
    u = u.strip()
    if not NAME_RE.match(u) or u.lower() == "background":
        raise HTTPException(400, "Use 1–24 letters, numbers, spaces, dots, dashes or underscores. \"background\" is reserved.")
    return u


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/state")
def state():
    entries = [{**e, "training": WLOG["trained"].get(e["entry_id"])} for e in stream_entries()]
    return {"model": be.name, "lam": LAM, "k": K, "temp": TEMP,
            "datastore_size": len(ds), "weak_match_threshold": WEAK_MATCH,
            "entries": entries, "users": stream_users(),
            "adapters": be.adapter_users(), "train_settings": be.train_settings,
            "weights_version": WLOG["version"], "history": WLOG["history"],
            "trace_available": bool(VIEWER), "viewer": VIEWER}


@app.post("/api/ingest")
def api_ingest(body: IngestIn):
    global TEMP
    if not body.consent:
        raise HTTPException(400, "Confirm that your text can be stored and shown to others.")
    u, text = check_name(body.user_id), body.text.strip()
    if not text or len(text) > MAX_CHARS:
        raise HTTPException(400, f"Write between 1 and {MAX_CHARS} characters.")
    with lock:
        eid = ingest_memory(u, text)
        TEMP = kc.calibrate_temp(ds)
        ds.save(DS_PATH, be.name)
        t0 = time.time()
        stats = be.train(u, [text])
        stats["seconds"] = time.time() - t0
        WLOG["trained"][eid] = stats
        record(f"{u} added text")
    return {"entry_id": eid, **state()}


@app.delete("/api/entries/{entry_id}")
def api_delete(entry_id: int):
    global TEMP
    with lock:
        e = next((e for e in ds.entries if e["entry_id"] == entry_id), None)
        if e is None or e["user_id"] == "background":
            raise HTTPException(404, "No such entry.")
        u = e["user_id"]
        ds.remove_entry(entry_id)
        TEMP = kc.calibrate_temp(ds)
        ds.save(DS_PATH, be.name)
        WLOG["trained"].pop(entry_id, None)
        # Weights can't subtract one text. Retrain this user's adapter from scratch on what's left.
        left = [x["text"] for x in stream_entries() if x["user_id"] == u]
        if left:
            stats = be.train(u, left, reset=True)
            for x in stream_entries():
                if x["user_id"] == u:
                    WLOG["trained"][x["entry_id"]] = {**stats, "retrained": True}
        else:
            be.remove_adapter(u)
        record(f"{u} removed text")
    return state()


@app.post("/api/ask")
def api_ask(body: AskIn):
    u, qtext = check_name(body.user_id), body.question.strip()
    if not qtext or len(qtext) > MAX_CHARS:
        raise HTTPException(400, f"Write a question between 1 and {MAX_CHARS} characters.")
    with lock:
        ids = be.prompt_ids(qtext)
        m_text, m_steps = generate_memory(ids, LAM, record_steps=True)
        active = be.adapter_users()
        w_text, w_steps = generate_weights(ids, active, record_steps=True)
        qid = len(QUERIES)
        QUERIES[qid] = {
            "query_id": qid, "user_id": u, "question": qtext, "ids": ids, "created": time.time(),
            "memory": {"answer": m_text, "steps": m_steps, "lam": LAM, "temp": TEMP,
                       "users_at_time": stream_users()},
            "weights": {"answer": w_text, "steps": w_steps, "adapters_at_time": active,
                        "weights_version": WLOG["version"]}}
    return public_query(QUERIES[qid])


@app.post("/api/regenerate")
def api_regen(body: RegenIn):
    q = QUERIES.get(body.query_id)
    if q is None:
        raise HTTPException(404, "No such answer.")
    ex = set(body.exclude_users)
    with lock:
        if body.mode == "weights":
            active = [] if body.base_only else [a for a in be.adapter_users() if a not in ex]
            text, _ = generate_weights(q["ids"], active)
        else:
            text, _ = generate_memory(q["ids"], 0.0 if body.base_only else LAM, ex)
    return {"text": text, **body.model_dump(),
            "note": "Uses the stream and adapters as they are now, which may differ from when the answer was made."}


@app.post("/api/weights_counterfactual")
def api_weights_cf(body: TokenIn):
    q = QUERIES.get(body.query_id)
    if q is None or not (0 <= body.pos < len(q["weights"]["steps"])):
        raise HTTPException(404, "No such word.")
    s = q["weights"]["steps"][body.pos]
    ids, t = s["_ids"], s["token_id"]
    with lock:
        users = be.adapter_users()
        def run(active):
            be.set_active(active)
            _, p = be.step(ids)
            top = int(np.argmax(p))
            return {"p_chosen": float(p[t]), "top_text": be.decode([top]), "flips": top != t}
        out = {"all": run(users), "base": run([]), "per_user": {}}
        for u in users:
            out["per_user"][u] = {"without": run([a for a in users if a != u]), "alone": run([u])}
    out["stale"] = WLOG["version"] != q["weights"]["weights_version"]
    out["adapters_now"] = users
    return out


@app.post("/api/trace")
def api_trace(body: TokenIn):
    if not VIEWER:
        raise HTTPException(400, "Attribution graphs aren't available in this mode.")
    q = QUERIES.get(body.query_id)
    if q is None or not (0 <= body.pos < len(q["memory"]["steps"])):
        raise HTTPException(404, "No such word.")
    slug = f"q{body.query_id}-t{body.pos}"
    if slug not in TRACES:
        tr = q["memory"]["steps"][body.pos]["_trace"]
        TRACES[slug] = {"slug": slug, "status": "queued", "queued": time.time(), **tr}
        trace_q.put(slug)
    return trace_status(slug)


@app.get("/api/trace/{slug}")
def trace_status(slug: str):
    j = TRACES.get(slug)
    if j is None:
        raise HTTPException(404, "No such trace.")
    out = {k: v for k, v in j.items() if k != "prefix_ids"}
    out["queue_position"] = list(trace_q.queue).index(slug) + 1 if slug in trace_q.queue else 0
    if j["status"] == "done":
        out["url"] = f"{VIEWER}?slug={slug}"
    return out


if __name__ == "__main__":
    import uvicorn
    print(f"Open http://localhost:{PORT}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT)
