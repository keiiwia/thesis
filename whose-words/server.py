"""Local server: the same stream feeds two learners.

Memory: a kNN datastore over the frozen base model. Provenance is exact.
Weights: one LoRA adapter per user, trained on their words. Real learning, real forgetting,
         and provenance that can only be estimated by switching adapters off.

Run:  uv run python server.py          (Gemma-3 270M IT on CPU)
"""
import os, re, json, math, random, shutil, threading, queue, time, traceback
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import knn_core as kc
import address as addr
from physical import cipher_wheel as cw

HERE = os.path.dirname(os.path.abspath(__file__))
LAM = float(os.environ.get("LAM", 0.25))
K = int(os.environ.get("K", 16))
# The stream may only vote among the words the untouched model finds fluent at each step (its top-p nucleus).
# Without this, a high LAM lets stored fragments override grammar and replies fall apart. 0 turns it off.
STREAM_TOP_P = float(os.environ.get("STREAM_TOP_P", 0.9))
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
FOUND_PATH = os.path.join(DATA, "found.json")
FEEDBACK_PATH = os.path.join(DATA, "feedback.json")   # UI annotations sent from the Agentation toolbar
# The library: mutated copies of what people wrote, credited to no one. Stands in for Borges'
# Library of Babel near the stream: every text that differs from yours by a few letters is on a shelf.
LIB = "library"
BABEL_ALPHABET = "abcdefghijklmnopqrstuvwxyz ,."   # the Library's 29 characters (Borges gives 25)
RESERVED = {"background", LIB}

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


def calibrate():
    # People's texts set the distance scale. The library's near-copies would shrink it.
    return kc.calibrate_temp(ds, exclude_users={LIB})


TEMP = calibrate()
TEXTS = {e["entry_id"]: e for e in ds.entries}


def stream_entries():
    return sorted((e for e in ds.entries if e["user_id"] not in RESERVED), key=lambda e: e["entry_id"])


def stream_users():
    return [u for u in ds.users() if u not in RESERVED]


def library_info():
    lib = [e for e in ds.entries if e["user_id"] == LIB]
    return {"texts": len(lib), "positions": int(ds.user_mask({LIB}).sum()),
            "copies": LIBRARY.get("copies", 0), "rate": LIBRARY.get("rate", 0.0)}


def mutate(text, rate, rng):
    return "".join(rng.choice(list(BABEL_ALPHABET)) if rng.random() < rate else c for c in text)


LIBRARY = next((e.get("library") for e in ds.entries if e["user_id"] == LIB and e.get("library")), None) or {}


# ---------- found: credit to whoever searched ----------
# In the Library of Babel nobody writes; every text is already shelved, and the only act is finding it.
# Here each memory answer logs how many words' worth of each writer's text the asker's question pulled out.
FOUND = json.load(open(FOUND_PATH)) if os.path.exists(FOUND_PATH) else []


def log_found(asker, question, steps):
    pulled = {}
    for st in steps:
        for u, v in st["shares"].items():
            if u != "parametric":
                pulled[u] = pulled.get(u, 0.0) + v
    FOUND.append({"t": time.time(), "finder": asker, "question": question,
                  "words": len(steps), "pulled": pulled})
    json.dump(FOUND, open(FOUND_PATH, "w"))


def found_summary():
    people = {}
    def row(u):
        return people.setdefault(u, {"found": 0.0, "found_from_others": 0.0, "written_found": 0.0,
                                     "written_found_by_others": 0.0, "from": {}, "by": {}})
    for f in FOUND:
        for w, v in f["pulled"].items():
            a, b = row(f["finder"]), row(w)
            a["found"] += v; b["written_found"] += v
            a["from"][w] = a["from"].get(w, 0.0) + v
            b["by"][f["finder"]] = b["by"].get(f["finder"], 0.0) + v
            if w != f["finder"]:
                a["found_from_others"] += v; b["written_found_by_others"] += v
    return {"people": people, "searches": len(FOUND),
            "recent": [{k: f[k] for k in ("t", "finder", "question", "words", "pulled")} for f in FOUND[-12:]][::-1]}


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


# ---------- price: bits to reach a message ----------
# A text's cost in bits is -log2 of the probability of each of its tokens, summed: every bit halves the odds.
# In a library where every token is equally likely (Babel), each costs log2(vocabulary) bits.
def price_message(text):
    """What it costs to reach this text, priced before it joins the stream (so it can't pay for itself)."""
    H, T, p0, ok = be.reply_probs(text, top_p=STREAM_TOP_P)
    ok = ok or [None] * len(T)
    lib = {LIB}
    def stream_p(exclude):
        return np.array([kc.knn_token_prob(ds, h, p, t, LAM, K, TEMP, exclude, a) for h, p, t, a in zip(H, p0, T, ok)])
    bits = lambda p: -np.log2(np.maximum(p, 1e-300))
    pm = stream_p(lib)
    with_all = float(bits(pm).sum())
    # Leave-one-person-out: how many more bits the text costs without that person's words.
    # These don't add up to the total saving, because bits are logs of a sum.
    subsidy = {u: float(bits(stream_p(lib | {u})).sum()) - with_all for u in stream_users()}
    active = be.adapter_users()
    pw = be.reply_probs(text, active)[2] if active else p0
    return {"tokens": len(T), "babel_per_token": math.log2(len(be.tok)),
            "babel": len(T) * math.log2(len(be.tok)), "alone": float(bits(p0).sum()),
            "stream": with_all, "weights": float(bits(pw).sum()), "subsidy": subsidy,
            "per_token": [{"text": be.decode([t]), "alone": float(a), "stream": float(m), "weights": float(w)}
                          for t, a, m, w in zip(T, bits(p0), bits(pm), bits(pw))]}


# ---------- exchange: what a prompt buys ----------
def exchange(steps, spent):
    """The prompt's side of the trade. spent: bits to reach the prompt itself. bought: how many bits cheaper
    the memory reply is after this prompt than after an empty one (pointwise mutual information)."""
    gen = [s["token_id"] for s in steps]
    if not gen:
        return None
    H, p0, ok = be.continuation_probs(be.prompt_ids(""), gen, top_p=STREAM_TOP_P)
    without = [-math.log2(max(kc.knn_token_prob(ds, h, p, t, LAM, K, TEMP, {LIB}, a), 1e-300))
               for h, p, t, a in zip(H, p0, gen, ok or [None] * len(gen))]
    with_ = [-math.log2(max(s["p_final"], 1e-300)) for s in steps]
    return {"spent": spent, "with": sum(with_), "without": sum(without), "bought": sum(without) - sum(with_),
            "babel": len(gen) * math.log2(len(be.tok)),
            "per_token": [{"with": w, "without": o} for w, o in zip(with_, without)]}


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


def open_words(p, mass=0.95):
    """How many words it takes to cover 95% of the odds: the corridor's width at this step."""
    return int(np.searchsorted(np.cumsum(np.sort(p)[::-1]), mass) + 1)


def generate_memory(ids, lam, exclude=(), record_steps=False, library=False):
    be.set_active([])                      # memory always runs on the untouched base model
    ids, gen, steps = list(ids), [], []
    users = [u for u in stream_users() if u not in exclude]
    exclude = set(exclude) if library else set(exclude) | {LIB}
    for _ in range(MAX_NEW):
        h, p_lm = be.step(ids)
        ok = kc.plausible(p_lm, STREAM_TOP_P)
        st = kc.knn_step(ds, h, p_lm, lam, K, TEMP, exclude, ok)
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
                text = TEXTS[m["entry_id"]]["text"]
                s, e = m["char_start"], m["char_end"]
                neighbors.append({
                    "user_id": m["user_id"], "entry_id": m["entry_id"],
                    "value_text": text[s:e], "context_before": text[max(0, s - 120):s],
                    "context_after": text[e:e + 60], "weight": float(w),
                    "d2_over_temp": float(d2 / TEMP), "supports_chosen": bool(ds.values[i] == t),
                    "filtered": ok is not None and int(ds.values[i]) not in ok,
                    "value_id": int(ds.values[i])})
            cf = {}
            for u in users:
                pc, top, ptop = kc.knn_point(ds, h, p_lm, lam, K, TEMP, set(exclude) | {u}, t, lm_top, ok)
                cf[u] = {"p_chosen": pc, "top_text": be.decode([top]), "top_p": ptop, "flips": top != t}
            near = float(st["d2"][0] / TEMP) if len(st["d2"]) else None
            open95 = open_words(st["p_final"])
            steps.append({
                "pos": len(gen), "token_id": t, "text": piece, "p_final": p_t,
                "p_lm": float(p_lm[t]), "shares": shares, "knn_share": 1.0 - shares["parametric"],
                "final_top5": top_k(st["p_final"]), "lm_top5": top_k(p_lm),
                "entropy_final": kc.entropy(st["p_final"]), "entropy_lm": kc.entropy(p_lm),
                "entropy_knn": kc.knn_entropy(ds, st),
                "nearest_d2_over_temp": near, "open95": open95, "weak_match": near is not None and near > WEAK_MATCH,
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
                "p_tuned": float(p[t]), "p_base": float(p0[t]), "open95": open_words(p),
                # share of this word's probability the base model can't account for (0 if the base liked it more)
                "lift": max(0.0, 1.0 - float(p0[t]) / float(p[t])),
                "tuned_top5": top_k(p), "base_top5": top_k(p0), "_ids": list(ids)})
        gen.append(t); ids.append(t)
        if repeating(gen):
            break
    return be.decode(gen), steps


def strip_private(steps, keys):
    return [{k: v for k, v in s.items() if k not in keys} for s in steps]


def public_library(q):
    lib = q.get("library")
    return lib and {**{k: v for k, v in lib.items() if k != "steps"},
                    "steps": strip_private(lib["steps"], {"_trace"})}


def public_query(q):
    return {"query_id": q["query_id"], "user_id": q["user_id"], "question": q["question"],
            "created": q["created"], "library": public_library(q), "price": q["price"],
            "exchange": q.get("exchange"),
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
    log: bool = True           # false: don't credit this ask in the found ledger (exports, tests)


class RegenIn(BaseModel):
    query_id: int
    mode: str = "memory"            # "memory" or "weights"
    exclude_users: list[str] = []
    base_only: bool = False


class LibraryIn(BaseModel):
    copies: int = 10           # mutated copies of each person's text
    rate: float = 0.1          # chance that each character is swapped for a random one


class FeedbackIn(BaseModel):
    output: str                # Agentation's markdown: selectors, positions and comments
    annotations: list = []


class QueryIn(BaseModel):
    query_id: int


class OddsIn(BaseModel):
    query_id: int
    pos: int
    n: int = 12                # how many of each side's likeliest words to list


class BranchIn(BaseModel):
    query_id: int
    pos: int            # the word in the memory reply where the path turns off
    token_id: int       # the word taken there instead


class TokenIn(BaseModel):
    query_id: int
    pos: int


def check_name(u):
    u = u.strip()
    if not NAME_RE.match(u) or u.lower() in RESERVED:
        raise HTTPException(400, "Use 1–24 letters, numbers, spaces, dots, dashes or underscores. \"background\" and \"library\" are reserved.")
    return u


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/sketch")
def sketch():
    """A static design sketch of the calmer page, built from captured replies."""
    return FileResponse(os.path.join(HERE, "static", "sketch.html"))


def printable(title, sheets, bar):
    """Letter sheets (SVG) as one page to print at 100%, with a small bar that doesn't print."""
    return HTMLResponse(f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>
@page{{size:letter;margin:0}} html{{color-scheme:light;background:#fff;color:#000}} body{{margin:0;font:13px system-ui,sans-serif}} a{{color:#000}} .bar{{padding:8px 12px;border-bottom:1px solid #000;display:flex;gap:14px;align-items:center}}
.sheet svg{{display:block;width:8.5in;height:11in}} .sheet + .sheet{{break-before:page}} @media print{{.bar{{display:none}}}}</style></head>
<body><div class="bar">Print at 100% on Letter, no scaling. <button onclick="print()">Print</button> {bar}</div>
{"".join(f'<div class="sheet">{s}</div>' for s in sheets)}</body></html>""")


# ---------- moments: a reply's address, taken away on a USB stick ----------
# The stick doesn't carry the reply. It carries the reply's address under the stream as it was (a few Babel
# characters), a copy of that stream, and an offline page. The address only reads as the reply under that weighting.
MOMENTS = os.path.join(DATA, "moments")


def measure_stream(store):
    def m(ids):
        be.set_active([])
        h, p = be.step(ids)
        return kc.knn_step(store, h, p, LAM, K, TEMP, {LIB}, kc.plausible(p, STREAM_TOP_P))["p_final"]
    return m


def measure_alone(ids):
    be.set_active([])
    return be.step(ids)[1]


def page_step(s):
    """One reply word in the offline page's format. Other people's stored texts stay behind: only names and
    weights travel on the stick, not their words."""
    votes = {}
    for n in s["neighbors"]:
        if n["weight"] > 0:
            v = votes.setdefault(n["value_id"], {})
            v[n["user_id"]] = v.get(n["user_id"], 0.0) + n["weight"]
    pusher = lambda i: max(votes[i], key=votes[i].get) if i in votes else None
    lm = s["lm_top5"][0]
    return {"t": s["text"], "p": s["p_final"], "plm": s["p_lm"], "open": s["open95"], "sh": s["shares"],
            "doors": [[d["text"], d["p"], pusher(d["id"])] for d in s["final_top5"] if d["id"] != s["token_id"]][:3],
            "alone": lm["text"] if lm["id"] != s["token_id"] else None,
            "weak": s["weak_match"], "near": round(s.get("nearest_d2_over_temp") or 0, 1),
            "cf": {u: [c["p_chosen"], c["top_text"], c["flips"]] for u, c in s["counterfactual"].items()},
            "nb": [{"u": n["user_id"], "v": n["value_text"], "w": n["weight"], "ok": n["supports_chosen"],
                    "f": n.get("filtered", False)} for n in s["neighbors"]],
            "lm5": [[x["text"], x["p"]] for x in s["lm_top5"]], "fin5": [[x["text"], x["p"]] for x in s["final_top5"]]}


def build_moment(q):
    """Encode the reply's address under the stream as it is now, check it decodes back exactly, read it with the
    model alone, and write the folder that goes on the stick. Returns the folder's path."""
    steps, prefix = q["memory"]["steps"], q["ids"]
    tokens = [s["token_id"] for s in steps]
    with lock:
        X, k = addr.encode(measure_stream(ds), prefix, tokens)
        back = addr.decode(measure_stream(ds), prefix, X, k, len(tokens))
        alone = be.decode(addr.decode(measure_alone, prefix, X, k, len(tokens)))
        folder_name = time.strftime("whose-words %Y-%m-%d %H%M ") + re.sub(r"[^A-Za-z0-9 ._-]", "", q["user_id"])
        folder = os.path.join(MOMENTS, folder_name)
        os.makedirs(os.path.join(folder, "stream"), exist_ok=True)
        ds.save(os.path.join(folder, "stream", "datastore"), be.name)
        # The copy keeps what reading the address needs (vectors, word numbers, names), not people's texts.
        snap = os.path.join(folder, "stream", "datastore.json")
        info = json.load(open(snap))
        info["entries"] = [{k: v for k, v in e.items() if k != "text"} for e in info["entries"]]
        json.dump(info, open(snap, "w"))
    if back != tokens:
        raise HTTPException(500, "The address didn't decode back to the reply, so no moment was written.")
    reply = be.decode(tokens)
    address = addr.to_text(X, k)
    babelized = cw.babelize(reply)
    people = stream_users()
    moment = {"taken": time.strftime("%Y-%m-%d %H:%M"), "by": q["user_id"], "question": q["question"], "reply": reply,
              "address": address, "address_chars": k, "address_bits": k * math.log2(addr.BASE),
              "babel_chars": len(babelized), "babel_bits": len(babelized) * math.log2(addr.BASE),
              "readings": {"stream": reply, "alone": alone, "babel": address},
              "stream_people": len(people), "stream_messages": len(stream_entries()),
              # what it takes to read the address again at the installation
              "decode": {"model": be.name, "prefix_ids": prefix, "tokens": len(tokens), "lam": LAM, "k": K,
                         "temp": TEMP, "stream_top_p": STREAM_TOP_P, "quantise_total": addr.TOTAL}}
    X_ = q.get("exchange") or {}
    data = [{"q": q["question"], "who": q["user_id"], "steps": [page_step(s) for s in steps],
             "spent": X_.get("spent", 0), "bought": X_.get("bought", 0), "with": X_.get("with", 0),
             "without": X_.get("without", 0), "babel": X_.get("babel", 0),
             "crowd": q["weights"]["answer"], "adapters": q["weights"]["adapters_at_time"]}]
    tpl = open(os.path.join(HERE, "static", "moment.html")).read()
    safe = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    open(os.path.join(folder, "open me.html"), "w").write(tpl.replace("__DATA__", safe(data)).replace("__MOMENT__", safe(moment)))
    json.dump(moment, open(os.path.join(folder, "moment.json"), "w"), indent=1, ensure_ascii=False)
    open(os.path.join(folder, "read me.txt"), "w").write(
        f"whose words? · a moment taken {moment['taken']}\n\n"
        f"Open “open me.html” in any browser. It works offline.\n\n"
        f"This stick doesn't hold the reply. It holds the reply's address, “{address}”: {k} characters of the\n"
        f"Library of Babel's alphabet. Read with the stream as it was when you took it, the address gives back the reply\n"
        f"exactly. Read with the model alone, the same address reads as a different text. In Babel itself, with no\n"
        f"weighting at all, the address is only its own {k} characters; the reply's address there is the whole reply,\n"
        f"{len(babelized)} characters long.\n\n"
        f"“stream/” is a copy of the stream at that moment, in the model's own form (vectors, word numbers and names,\n"
        f"not people's texts): what makes the address legible. The page shows only names and how much each person\n"
        f"shaped each word.\n")
    return folder


class MomentIn(BaseModel):
    query_id: int
    volume: str = ""           # a drive under /Volumes to write to; empty: just build it for download


def volumes():
    """Removable drives: mount points under /Volumes other than the startup disk."""
    out = []
    for name in sorted(os.listdir("/Volumes")) if os.path.isdir("/Volumes") else []:
        path = os.path.join("/Volumes", name)
        try:
            if os.path.ismount(path) and not os.path.samefile(path, "/") and os.access(path, os.W_OK):
                free = shutil.disk_usage(path).free
                out.append({"name": name, "free_mb": round(free / 1e6)})
        except OSError:
            pass
    return out


@app.get("/api/volumes")
def api_volumes():
    return volumes()


@app.post("/api/moment")
def api_moment(body: MomentIn):
    q = QUERIES.get(body.query_id)
    if q is None:
        raise HTTPException(404, "No such reply. Replies are kept only while the server runs.")
    if body.volume and body.volume not in {v["name"] for v in volumes()}:
        raise HTTPException(400, "That drive isn't plugged in (or can't be written to).")
    folder = build_moment(q)
    moment = json.load(open(os.path.join(folder, "moment.json")))
    written = None
    if body.volume:
        dest = os.path.join("/Volumes", body.volume, os.path.basename(folder))
        shutil.copytree(folder, dest, dirs_exist_ok=True)
        written = dest
    return {"folder": os.path.basename(folder), "written_to": written, "download": "/moments/" + os.path.basename(folder) + ".zip",
            "address": moment["address"], "readings": moment["readings"], "babel_chars": moment["babel_chars"]}


@app.get("/moments/{name}.zip")
def moment_zip(name: str):
    folder = os.path.join(MOMENTS, os.path.basename(name))
    if not os.path.isdir(folder):
        raise HTTPException(404, "No such moment.")
    base = os.path.join(MOMENTS, "_zips", os.path.basename(name))
    os.makedirs(os.path.dirname(base), exist_ok=True)
    path = shutil.make_archive(base, "zip", MOMENTS, os.path.basename(name))
    return FileResponse(path, filename=os.path.basename(name) + ".zip")


@app.get("/wheel")
def wheel_kit():
    """The Babel cipher wheel: guide, outer disk, inner disk. One kit decodes every card."""
    k = cw.kit()
    return printable("Babel cipher wheel", [k["guide"], k["outer"], k["inner"]], "")


@app.get("/wheel/vellum")
def wheel_vellum(d: float = 0):
    """The station's inner disk, in real millimetres, to print on vellum. d: disk diameter in mm (default 90)."""
    d = d if 50 <= d <= 180 else 0
    return printable("Vellum inner disk", [cw.vellum_sheet(d or None)], f'<a href="/station?d={d or cw.STATION["disk_d"]:g}">open the station</a>')


@app.get("/station")
def station_page():
    return FileResponse(os.path.join(HERE, "static", "station.html"))


@app.get("/card/{query_id}")
def babel_card(query_id: int, key: int = 0):
    """Any memory reply as a Babel card for the wheel. key 1–28; anything else picks one at random."""
    q = QUERIES.get(query_id)
    if q is None:
        raise HTTPException(404, "No such reply. Replies are kept only while the server runs.")
    plain, cut = cw.fit(cw.babelize(q["memory"]["answer"]))
    key = key if 1 <= key < cw.N else random.randint(1, cw.N - 1)
    return printable(f"Babel card · key {key}", [cw.card(plain, key, q["question"], cut=cut)],
                     f'<a href="?key={key}">keep key {key}</a><a href="?">another key</a><a href="/wheel" target="_blank">the wheel</a>')


@app.get("/api/state")
def state():
    entries = [{**e, "training": WLOG["trained"].get(e["entry_id"])} for e in stream_entries()]
    return {"model": be.name, "lam": LAM, "k": K, "temp": TEMP,
            "datastore_size": len(ds), "weak_match_threshold": WEAK_MATCH,
            "entries": entries, "users": stream_users(),
            "adapters": be.adapter_users(), "train_settings": be.train_settings,
            "weights_version": WLOG["version"], "history": WLOG["history"],
            "trace_available": bool(VIEWER), "viewer": VIEWER,
            "library": library_info(), "found": found_summary(), "vocab": len(be.tok)}


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
        TEXTS[eid] = ds.entries[-1]
        TEMP = calibrate()
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
        if e is None or e["user_id"] in RESERVED:
            raise HTTPException(404, "No such entry.")
        u = e["user_id"]
        ds.remove_entry(entry_id)
        TEXTS.pop(entry_id, None)
        TEMP = calibrate()
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
        price = price_message(qtext)
        m_text, m_steps = generate_memory(ids, LAM, record_steps=True)
        active = be.adapter_users()
        w_text, w_steps = generate_weights(ids, active, record_steps=True)
        qid = len(QUERIES)
        QUERIES[qid] = {
            "query_id": qid, "user_id": u, "question": qtext, "ids": ids, "created": time.time(), "price": price,
            "memory": {"answer": m_text, "steps": m_steps, "lam": LAM, "temp": TEMP,
                       "users_at_time": stream_users()},
            "weights": {"answer": w_text, "steps": w_steps, "adapters_at_time": active,
                        "weights_version": WLOG["version"]}}
        QUERIES[qid]["exchange"] = exchange(m_steps, price["stream"])
        if body.log:
            log_found(u, qtext, m_steps)
    return public_query(QUERIES[qid])


def clear_library():
    for e in [e for e in ds.entries if e["user_id"] == LIB]:
        ds.remove_entry(e["entry_id"])
        TEXTS.pop(e["entry_id"], None)
    LIBRARY.clear()


@app.post("/api/library")
def api_library_fill(body: LibraryIn):
    if not (1 <= body.copies <= 50 and 0.0 < body.rate <= 1.0):
        raise HTTPException(400, "Use 1–50 copies and a swap rate above 0 and at most 1.")
    rng = np.random.default_rng(0)
    with lock:
        clear_library()
        LIBRARY.update(copies=body.copies, rate=body.rate)
        for e in stream_entries():
            for _ in range(body.copies):
                t = mutate(e["text"], body.rate, rng)
                eid = ds.add_entry(LIB, t, *be.encode_ingest(t),
                                   extra={"created": time.time()})
                ds.entries[-1]["library"] = dict(LIBRARY)   # remembers the settings across restarts
                TEXTS[eid] = ds.entries[-1]
        ds.save(DS_PATH, be.name)
    return state()


@app.delete("/api/library")
def api_library_clear():
    with lock:
        clear_library()
        ds.save(DS_PATH, be.name)
    return state()


@app.post("/api/feedback")
def api_feedback(body: FeedbackIn):
    """Keep design feedback from the Agentation toolbar (?feedback in the page URL) for whoever edits the UI."""
    log = json.load(open(FEEDBACK_PATH)) if os.path.exists(FEEDBACK_PATH) else []
    log.append({"t": time.time(), "output": body.output, "annotations": body.annotations})
    json.dump(log, open(FEEDBACK_PATH, "w"), indent=1)
    return {"saved": len(body.annotations), "total_sends": len(log)}


@app.post("/api/library_answer")
def api_library_answer(body: QueryIn):
    """The same question, answered with the library's shelves open alongside the stream."""
    q = QUERIES.get(body.query_id)
    if q is None:
        raise HTTPException(404, "No such answer.")
    if not library_info()["texts"]:
        raise HTTPException(400, "The library is empty. Fill it first.")
    with lock:
        text, steps = generate_memory(q["ids"], LAM, record_steps=True, library=True)
        q["library"] = {"answer": text, "steps": steps, "lam": LAM, "temp": TEMP,
                        "users_at_time": stream_users(), **library_info()}
    return public_library(q)


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


@app.post("/api/branch")
def api_branch(body: BranchIn):
    """Walk through a side door: keep the memory reply up to pos, take token_id there, and let it run on."""
    q = QUERIES.get(body.query_id)
    if q is None or not (0 <= body.pos < len(q["memory"]["steps"])) or not (0 <= body.token_id < len(be.tok)):
        raise HTTPException(404, "No such word.")
    kept = [s["token_id"] for s in q["memory"]["steps"][:body.pos]]
    with lock:
        text, steps = generate_memory(q["ids"] + kept + [body.token_id], LAM, record_steps=True)
    return {"kept": be.decode(kept), "door": piece_of(kept, body.token_id), "text": text, "pos": body.pos,
            "token_id": body.token_id, "steps": strip_private(steps, {"_trace"}),
            "note": "Uses the stream as it is now, which may differ from when the reply was made."}


@app.post("/api/odds")
def api_odds(body: OddsIn):
    """The odds at one word of a memory reply, model alone and with the stream, for the likeliest words on
    either side (for the printed odds wheels in physical/). Uses the stream as it is now."""
    q = QUERIES.get(body.query_id)
    if q is None or not (0 <= body.pos < len(q["memory"]["steps"])):
        raise HTTPException(404, "No such word.")
    step = q["memory"]["steps"][body.pos]
    with lock:
        be.set_active([])
        h, p_lm = be.step(step["_trace"]["prefix_ids"])
        st = kc.knn_step(ds, h, p_lm, LAM, K, TEMP, {LIB}, kc.plausible(p_lm, STREAM_TOP_P))
    pf, n = st["p_final"], max(1, min(body.n, 40))
    votes = {}
    for i, w in zip(st["idx"], st["w"]):
        if w > 0:
            v = votes.setdefault(int(ds.values[i]), {})
            u = ds.meta[i]["user_id"]
            v[u] = v.get(u, 0.0) + float(w)
    ids = set(np.argsort(-p_lm)[:n].tolist()) | set(np.argsort(-pf)[:n].tolist()) | {step["token_id"]}
    words = [{"id": int(i), "text": be.decode([int(i)]), "alone": float(p_lm[i]), "stream": float(pf[i]),
              "votes": votes.get(int(i), {})} for i in ids]
    return {"question": q["question"], "pos": body.pos, "chosen": step["token_id"], "vocab": len(be.tok),
            "context": "".join(s["text"] for s in q["memory"]["steps"][:body.pos]),
            "words": sorted(words, key=lambda w: -w["alone"])}


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
