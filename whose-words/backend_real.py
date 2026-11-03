"""Gemma-3 270M (instruction-tuned) on CPU, with one LoRA adapter per user.

Memory mode runs the base model (every adapter off) plus the kNN datastore.
Weights mode runs the base model plus users' adapters, no datastore.
"""
import math, os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-270m-it")
# Not verified by download: named by analogy with the 1B/4B repos used in circuit-tracer's tests.
# If loading fails, find the 270M IT entry in huggingface.co/collections/mwhanna/gemma-scope-2-transcoders-circuit-tracer
TRANSCODERS = os.environ.get(
    "TRANSCODERS", "mwhanna/gemma-scope-2-270m-it/transcoder_all/width_16k_l0_small_affine")
DEVICE = os.environ.get("DEVICE", "cpu")
INGEST_PROMPT = "Share your view."  # user text is stored and trained on as the model's reply to this

LORA_R = int(os.environ.get("LORA_R", 8))
LORA_ALPHA = float(os.environ.get("LORA_ALPHA", 16))
TRAIN_STEPS = int(os.environ.get("TRAIN_STEPS", 30))   # optimizer steps per new text
LR = float(os.environ.get("LR", 5e-4))
# "mean": switched-on adapters are averaged, so thirteen people don't push thirteen times as hard. "sum": added at full strength.
ADAPTER_MIX = os.environ.get("ADAPTER_MIX", "mean")
# Attention projections only. The MLPs stay untouched, so they still match the base model.
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")

GRAD_NOTE = "inference_mode is set per call: grad mode is per-thread and requests run on worker threads"


class MultiLoRALinear(nn.Module):
    """A frozen Linear plus any number of named low-rank adapters, switchable per call."""

    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.A, self.B = nn.ParameterDict(), nn.ParameterDict()
        self.active = []
        self.mix = 1.0          # each active adapter's weight in the blend

    def add(self, key):
        dev = self.base.weight.device
        A = torch.empty(self.r, self.base.in_features, device=dev)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        self.A[key] = nn.Parameter(A)
        self.B[key] = nn.Parameter(torch.zeros(self.base.out_features, self.r, device=dev))

    def drop(self, key):
        del self.A[key]
        del self.B[key]

    def forward(self, x):
        y = self.base(x)
        for k in self.active:
            y = y + (x @ self.A[k].t()) @ self.B[k].t() * (self.scale * self.mix)
        return y


class Backend:
    supports_trace = True

    def __init__(self):
        self.name = MODEL_ID
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32, attn_implementation="eager").to(DEVICE).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.loras = []
        for mod in list(self.model.modules()):
            for t in TARGETS:
                child = getattr(mod, t, None)
                if isinstance(child, nn.Linear):
                    w = MultiLoRALinear(child, LORA_R, LORA_ALPHA)
                    setattr(mod, t, w)
                    self.loras.append(w)
        assert self.loras, "No attention projections found to attach adapters to."
        self.keys, self._n = {}, 0      # user -> parameter key (keys can't contain dots or spaces)
        self.softcap = getattr(self.model.config, "final_logit_softcapping", None)
        self.dim = self.model.config.hidden_size
        self.stop_ids = {self.tok.convert_tokens_to_ids("<end_of_turn>"), self.tok.eos_token_id}
        self._rm = None
        self.train_settings = {"rank": LORA_R, "alpha": LORA_ALPHA, "steps_per_text": TRAIN_STEPS,
                               "lr": LR, "layers": list(TARGETS), "trained": "alone, no end marker"}

    # ---------- shared ----------
    def _prefix(self, user_msg):
        return self.tok.apply_chat_template([{"role": "user", "content": user_msg}],
                                            add_generation_prompt=True, tokenize=False)

    def _ids(self, s):
        return self.tok(s, add_special_tokens=False)["input_ids"]

    @torch.inference_mode()  # see GRAD_NOTE
    def _hidden(self, ids):
        out = self.model.model(input_ids=torch.tensor([ids], device=DEVICE))
        return out.last_hidden_state[0]  # post final norm: exactly what lm_head reads

    @torch.inference_mode()  # see GRAD_NOTE
    def encode_ingest(self, text):
        prefix = self._prefix(INGEST_PROMPT)
        enc = self.tok(prefix + text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        H = self._hidden(ids).float().cpu().numpy()
        L = len(prefix)
        keys, vals, spans = [], [], []
        for i in range(len(ids) - 1):
            s, e = offs[i + 1]
            if s >= L:  # only positions whose next token is part of the user's text
                keys.append(H[i]); vals.append(ids[i + 1]); spans.append((s - L, e - L))
        return np.array(keys, dtype=np.float32), vals, spans

    @torch.inference_mode()  # see GRAD_NOTE
    def continuation_probs(self, prefix_ids, target_ids, users=(), top_p=None, cap=100):
        """With users' adapters on: the hidden state before each target token, the probability the model gave it,
        and (with top_p) the set of words the model found fluent there, as in knn_core.plausible."""
        self.set_active(list(users))
        P, T = list(prefix_ids), list(target_ids)
        H = self._hidden(P + T)[len(P) - 1:-1]
        logits = self.model.lm_head(H).float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        probs = torch.softmax(logits, -1)
        p = probs.gather(1, torch.tensor(T, device=DEVICE)[:, None])[:, 0]
        allowed = None
        if top_p:
            v, i = probs.topk(cap, -1)
            n = (torch.cumsum(v, -1) < top_p).sum(-1) + 1
            allowed = [set(i[r, :int(n[r])].tolist()) for r in range(len(T))]
        self.set_active([])
        return H.float().cpu().numpy(), p.double().cpu().numpy(), allowed

    def reply_probs(self, text, users=(), top_p=None):
        """Same, for text as a reply to INGEST_PROMPT (how people's words are stored). Also returns its tokens."""
        T = self._ids(text)
        H, p, allowed = self.continuation_probs(self._ids(self._prefix(INGEST_PROMPT)), T, users, top_p)
        return H, T, p, allowed

    def prompt_ids(self, question):
        return self._ids(self._prefix(question))

    @torch.inference_mode()  # see GRAD_NOTE
    def step(self, ids):
        h = self._hidden(ids)[-1]
        logits = self.model.lm_head(h).float()
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return h.float().cpu().numpy(), torch.softmax(logits, -1).double().cpu().numpy()

    def decode(self, ids):
        return self.tok.decode(ids)

    # ---------- adapters ----------
    def adapter_users(self):
        return list(self.keys)

    def set_active(self, users):
        ks = [self.keys[u] for u in users if u in self.keys]
        mix = 1.0 / len(ks) if ks and ADAPTER_MIX == "mean" else 1.0
        for l in self.loras:
            l.active, l.mix = ks, mix

    def remove_adapter(self, user):
        k = self.keys.pop(user, None)
        if k is not None:
            for l in self.loras:
                l.drop(k)

    def train(self, user, texts, reset=False):
        """Continue training this user's adapter on texts, with only that adapter switched on.
        Each adapter learns one person's words from the untouched model, so "only X" means X alone,
        and blending them at answer time doesn't compound everyone's training."""
        if reset:
            self.remove_adapter(user)
        if user not in self.keys:
            self.keys[user] = f"u{self._n}"; self._n += 1
            for l in self.loras:
                l.add(self.keys[user])
        k = self.keys[user]
        params = [l.A[k] for l in self.loras] + [l.B[k] for l in self.loras]
        self.set_active([user])
        prefix_len = len(self._ids(self._prefix(INGEST_PROMPT)))
        examples = []
        for t in texts:
            # No end-of-reply marker: a person's message isn't a whole answer to every question, and training
            # on the marker taught blended adapters to stop after a few words.
            ids = self._ids(self._prefix(INGEST_PROMPT) + t)
            labels = [-100] * prefix_len + ids[prefix_len:]
            examples.append((torch.tensor([ids], device=DEVICE), torch.tensor([labels], device=DEVICE)))
        opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
        losses = []
        n = TRAIN_STEPS * len(examples)
        with torch.enable_grad():
            for s in range(n):
                ids, labels = examples[s % len(examples)]
                loss = self.model(input_ids=ids, labels=labels).loss
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                losses.append(loss.item())
        return {"steps": n, "loss_start": losses[0], "loss_end": losses[-1]}

    @torch.inference_mode()  # see GRAD_NOTE
    def score(self, question, answer):
        """Mean log-probability per token of a reference answer, under the current adapters."""
        p_ids, a_ids = self._ids(self._prefix(question)), self._ids(answer)
        logits = self.model(input_ids=torch.tensor([p_ids + a_ids], device=DEVICE)).logits[0].float()
        lp = torch.log_softmax(logits[len(p_ids) - 1:-1], -1)
        return float(lp.gather(1, torch.tensor(a_ids, device=DEVICE)[:, None]).mean())

    def save_adapters(self, path):
        state = {u: [(l.A[k].detach().cpu(), l.B[k].detach().cpu()) for l in self.loras]
                 for u, k in self.keys.items()}
        torch.save({"model": MODEL_ID, "settings": self.train_settings, "order": list(self.keys),
                    "state": state}, path)

    def load_adapters(self, path):
        """Returns the list of users restored, or None if nothing usable was saved."""
        if not os.path.exists(path):
            return None
        blob = torch.load(path, weights_only=False)
        if blob.get("model") != MODEL_ID or blob.get("settings") != self.train_settings:
            return None
        for u in blob["order"]:
            self.keys[u] = f"u{self._n}"; self._n += 1
            for l, (A, B) in zip(self.loras, blob["state"][u]):
                l.add(self.keys[u])
                l.A[self.keys[u]].data.copy_(A); l.B[self.keys[u]].data.copy_(B)
        return blob["order"]

    # ---------- attribution graphs (base model only) ----------
    def trace(self, prefix_ids, target_ids, slug, graph_dir, pt_dir):
        from circuit_tracer import ReplacementModel, attribute
        from circuit_tracer.utils import create_graph_files
        if self._rm is None:
            # Loaded fresh from the hub, so it never includes any user's adapter.
            self._rm = ReplacementModel.from_pretrained(
                MODEL_ID, TRANSCODERS, backend="nnsight",
                device=torch.device(DEVICE), dtype=torch.float32)
        g = attribute(prompt=list(prefix_ids), model=self._rm,
                      attribution_targets=torch.tensor(target_ids),
                      batch_size=64, max_feature_nodes=4096, verbose=False)
        pt = os.path.join(pt_dir, f"{slug}.pt")
        g.to_pt(pt)
        create_graph_files(graph_or_path=pt, slug=slug, output_path=graph_dir,
                           node_threshold=0.8, edge_threshold=0.98)
