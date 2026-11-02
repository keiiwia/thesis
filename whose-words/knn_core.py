"""kNN-LM core. Pure numpy so the math is testable without a GPU.

p_final(w) = (1 - lam) * p_LM(w) + lam * p_kNN(w)
p_kNN(w)   = sum over retrieved neighbors i with value_i == w of softmax(-d_i / temp)

Because p_final is a sum, each token's probability decomposes exactly into
a parametric share plus one share per contributing user.
"""
import json, os
import numpy as np


class Datastore:
    def __init__(self, dim):
        self.dim = dim
        self.keys = np.zeros((0, dim), dtype=np.float32)
        self.values = np.zeros((0,), dtype=np.int64)
        self.meta = []  # one dict per entry: user_id, entry_id, char_start, char_end
        self.entries = []  # one dict per ingested text: entry_id, user_id, text
        self.next_id = 0

    def __len__(self):
        return len(self.values)

    def add_entry(self, user_id, text, keys, values, char_spans, extra=None):
        """keys[i] is the hidden state at position i; values[i] is the token that came next."""
        assert len(keys) == len(values) == len(char_spans)
        entry_id = self.next_id
        self.next_id += 1
        self.entries.append({"entry_id": entry_id, "user_id": user_id, "text": text,
                             "created": extra.get("created") if extra else None})
        self.keys = np.vstack([self.keys, np.asarray(keys, dtype=np.float32)])
        self.values = np.concatenate([self.values, np.asarray(values, dtype=np.int64)])
        for s, e in char_spans:
            self.meta.append({"user_id": user_id, "entry_id": entry_id,
                              "char_start": int(s), "char_end": int(e)})
        return entry_id

    def remove_entry(self, entry_id):
        keep = np.array([m["entry_id"] != entry_id for m in self.meta], dtype=bool)
        self.keys, self.values = self.keys[keep], self.values[keep]
        self.meta = [m for m, k in zip(self.meta, keep) if k]
        before = len(self.entries)
        self.entries = [e for e in self.entries if e["entry_id"] != entry_id]
        return len(self.entries) < before

    def save(self, path_prefix, model_name):
        # Write both files to temporary names first so a crash can't leave them out of step.
        with open(path_prefix + ".tmp.npz", "wb") as f:
            np.savez(f, keys=self.keys, values=self.values)
        with open(path_prefix + ".tmp.json", "w") as f:
            json.dump({"model": model_name, "dim": self.dim, "next_id": self.next_id,
                       "meta": self.meta, "entries": self.entries}, f)
        os.replace(path_prefix + ".tmp.npz", path_prefix + ".npz")
        os.replace(path_prefix + ".tmp.json", path_prefix + ".json")

    @classmethod
    def load(cls, path_prefix, model_name):
        """Returns None if no saved datastore exists or it was built with a different model."""
        if not (os.path.exists(path_prefix + ".npz") and os.path.exists(path_prefix + ".json")):
            return None
        info = json.load(open(path_prefix + ".json"))
        if info["model"] != model_name:
            return None
        arr = np.load(path_prefix + ".npz")
        ds = cls(info["dim"])
        ds.keys, ds.values = arr["keys"].astype(np.float32), arr["values"].astype(np.int64)
        ds.meta, ds.entries, ds.next_id = info["meta"], info["entries"], info["next_id"]
        return ds

    def users(self):
        return sorted({m["user_id"] for m in self.meta})

    def search(self, query, k, exclude_users=()):
        """Exact L2 search. Fine for thousands of entries; swap in FAISS past ~1e5."""
        if len(self) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32)
        d2 = ((self.keys - q) ** 2).sum(axis=1)
        if exclude_users:
            mask = np.array([m["user_id"] in exclude_users for m in self.meta])
            d2 = np.where(mask, np.inf, d2)
        k = min(k, int(np.isfinite(d2).sum()))
        if k == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        idx = np.argpartition(d2, k - 1)[:k]
        idx = idx[np.argsort(d2[idx])]
        return idx, d2[idx]


def neighbor_weights(d2, temp):
    if len(d2) == 0:
        return np.zeros(0)
    z = -d2 / temp
    z = z - z.max()
    w = np.exp(z)
    return w / w.sum()


def knn_step(ds, query, p_lm, lam, k, temp, exclude_users=()):
    """One decoding step. Returns the full interpolated distribution plus neighbor details."""
    idx, d2 = ds.search(query, k, exclude_users)
    w = neighbor_weights(d2, temp)
    lam_eff = lam if len(idx) > 0 else 0.0
    p_final = (1.0 - lam_eff) * np.asarray(p_lm, dtype=np.float64)
    if len(idx):
        np.add.at(p_final, ds.values[idx], lam_eff * w)
    return {"p_final": p_final, "idx": idx, "d2": d2, "w": w, "lam_eff": lam_eff}


def decompose(ds, step, p_lm, token):
    """Exact split of p_final(token) into parametric + per-user shares (sums to 1)."""
    lam = step["lam_eff"]
    total = step["p_final"][token]
    parts = {"parametric": (1.0 - lam) * float(p_lm[token])}
    for i, wi in zip(step["idx"], step["w"]):
        if ds.values[i] == token:
            u = ds.meta[i]["user_id"]
            parts[u] = parts.get(u, 0.0) + lam * float(wi)
    return {key: v / total for key, v in parts.items()}, float(total)


def calibrate_temp(ds, n_sample=500, seed=0):
    """Median squared distance from an entry to its nearest *other* entry.

    Heuristic, not from the kNN-LM paper: the paper used raw distances with its own
    model's key scale. Gemma's hidden-state scale differs, so we set temp to a typical
    nearest-neighbor distance. Tune it and watch how peaked the weights get.
    """
    n = len(ds)
    if n < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    sample = rng.choice(n, size=min(n_sample, n), replace=False)
    nn = []
    for i in sample:
        d2 = ((ds.keys - ds.keys[i]) ** 2).sum(axis=1)
        d2[i] = np.inf
        nn.append(d2.min())
    return float(np.median(nn))


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def knn_point(ds, query, p_lm, lam, k, temp, exclude_users, token, lm_top):
    """Like knn_step but without building the full vocab-sized distribution.

    Returns p_final(token) and the argmax token. The argmax must be the LM's own top token
    or one of the retrieved neighbors' tokens, since only those can beat the LM's top token.
    """
    idx, d2 = ds.search(query, k, exclude_users)
    w = neighbor_weights(d2, temp)
    lam_eff = lam if len(idx) > 0 else 0.0
    boost = {}
    for i, wi in zip(idx, w):
        v = int(ds.values[i])
        boost[v] = boost.get(v, 0.0) + lam_eff * float(wi)
    def p(t):
        return (1.0 - lam_eff) * float(p_lm[t]) + boost.get(t, 0.0)
    cands = set(boost) | {int(lm_top)}
    top = max(cands, key=p)
    return p(int(token)), top, p(top)
