"""c24 -- depth x width scaling of the fixnorm Pareto exchange, split-metric.

Why this experiment is SPLIT
----------------------------
c23 answered the capacity question at L=2 across d in {32, 64} and found a flat
~1.8% perplexity penalty against a gap gain that grew 83x -> 879x. The obvious
next move is to push L and d together. A pre-flight sweep of the proposed grid
(random init, c18 methodology, results/c24_feasibility.json) showed that move is
only half well-posed:

    L=2  d=64   std gap 7.8e2   fixnorm 1.16
    L=2  d=512  std gap 3.4e11  fixnorm 3.2e3
    L=4  d=256  std gap 6.1e46  fixnorm 1.0e9
    L=8  d=64   std gap NaN     fixnorm 1.6e4
    L=12 d=256  std gap NaN     fixnorm NaN

The standard variant overflows float64 at L>=8 and fixnorm follows at L=12,
d>=256. Two consequences drive the design of this file:

  * PERPLEXITY has no ceiling and no overflow. It is measurable on the whole
    L x d grid and it is the half a GPU actually accelerates. Sweep it fully.
  * The RELAXATION GAP stops carrying certification meaning long before the grid
    ends. A gap of 1e9 says the certified bound is a billion times wider than
    the reachable set; a gain ratio between two such numbers is arithmetic, not
    evidence. Verify only where the bound still means something, and report the
    CEILING as the finding rather than extrapolating a trend through NaN.

So: train everywhere, verify on a subgrid, and never quote a gain ratio whose
denominator is above GAP_MEANINGFUL.

Division of hardware
--------------------
Training runs on CUDA when available (Colab T4 via the WSL colab CLI).
Verification stays on the unchanged NumPy engine in src/bounds.py, on CPU, and
deliberately so. Measured on the T4: FP64 0.252 TFLOP/s against FP32 3.67, a
14.5x penalty on its own silicon. The zonotope pass is float64 and small-matrix
bound; it is faster on this workstation's CPU than on a 2-vCPU cloud VM, and
porting it to CUDA would invalidate the "src/bounds.py unmodified" property that
every result from c14 onward rests on. Weights cross the boundary as float64.

This file is deliberately self-contained on the corpus side. c23_language.py
parses argv and loads the corpus at module scope, so importing it would hijack
this script's arguments -- the same defect c18_variants.py had -- and it is
hash-locked in the verified bundle. The corpus handling below is replicated
EXACTLY (same split, same val seed, same SEQ_LEN); --selfcheck asserts that this
file reproduces c23's published numbers at the shared L=2 cells.
"""
import sys, os, json, time, argparse, hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "tinyshakespeare.txt")

SEQ_LEN = 8          # unchanged from c23; this sweep moves L and d, not context
EPS = 1e-5

# A bound this many times wider than the attained width is not a certificate.
# Cells above it are still recorded, but are reported as CEILING EXCEEDED and
# are excluded from every gain ratio.
GAP_MEANINGFUL = 1e3


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=200)
    # Depth diverges without this. Measured at L=12/d=512, lr=3e-3, warmup=200:
    # all 3 seeds blew up (train loss 3.9e12, val ppl NaN). Default is 0 so the
    # cells already computed at the original recipe keep their fingerprint;
    # deep cells are run explicitly with --clip.
    ap.add_argument("--clip", type=float, default=0.0,
                    help="max grad norm; 0 disables clipping")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--widths", default="64,128,256,512")
    ap.add_argument("--layers", default="2,4,8,12")
    ap.add_argument("--variants", default="standard,fixnorm")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-seqs", type=int, default=16384)
    ap.add_argument("--rho", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache", default=os.path.join(ROOT, "results", "c24_cache"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "c24.json"))
    ap.add_argument("--train-only", action="store_true",
                    help="train + save weights, skip the NumPy verification pass")
    ap.add_argument("--verify-only", action="store_true",
                    help="verify cached weights; no training, no CUDA needed")
    ap.add_argument("--verify-layers", default="2,4",
                    help="subgrid where the gap is still meaningful")
    ap.add_argument("--verify-widths", default="64,128,256")
    ap.add_argument("--selfcheck", action="store_true",
                    help="reproduce c23's L=2 cells and assert agreement")
    return ap.parse_args(argv)


# --------------------------------------------------------------------- corpus
def load_corpus():
    raw = open(DATA, "rb").read()
    text = raw.decode("utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int64)
    n_train = int(0.9 * len(ids))
    return {"sha256": hashlib.sha256(raw).hexdigest(), "chars": chars,
            "vocab": len(chars), "ids": ids, "n_train": n_train,
            "train": ids[:n_train], "val": ids[n_train:], "n_chars": len(text)}


def sample(n, rng, ids):
    """n windows of SEQ_LEN+1 chars -> (context, next-char targets)."""
    hi = len(ids) - SEQ_LEN - 1
    off = rng.integers(0, hi, size=n)
    w = off[:, None] + np.arange(SEQ_LEN + 1)[None, :]
    win = ids[w]
    return win[:, :-1], win[:, 1:]


# ---------------------------------------------------------------------- model
class Norm(nn.Module):
    """standard vs fixnorm differ in ONE line. Replicated from c19.Norm so this
    file does not import a module that parses argv at import time."""

    def __init__(self, d, mode):
        super().__init__()
        self.mode = mode
        self.g = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.register_buffer("scale", torch.ones(1))
        self.calibrating = False

    def forward(self, x):
        xc = x - x.mean(-1, keepdim=True)
        s_data = torch.sqrt(xc.pow(2).mean(-1, keepdim=True) + EPS)
        if self.calibrating:
            with torch.no_grad():
                self.scale.fill_(float(s_data.mean()))
        s = s_data if self.mode == "standard" else self.scale
        return xc / s * self.g + self.b


class Block(nn.Module):
    """Pre-norm attention + MLP, matching c19.Block's structure and c18.block's
    abstract counterpart. Kept in lockstep with export_W below."""

    def __init__(self, d, n_heads, mode, act="relu"):
        super().__init__()
        self.h, self.dh, self.act = n_heads, d // n_heads, act
        self.n1, self.n2 = Norm(d, mode), Norm(d, mode)
        self.WQ = nn.Linear(d, d, bias=False); self.WK = nn.Linear(d, d, bias=False)
        self.WV = nn.Linear(d, d, bias=False); self.WO = nn.Linear(d, d, bias=False)
        self.fc_in = nn.Linear(d, 4 * d); self.fc_out = nn.Linear(4 * d, d)

    def attn(self, x):
        # Causal mask is load-bearing: without it the model reads the future and
        # perplexity is not a language-modelling number at all. It also has to
        # match src/bounds.py's softmax_interval, which masks the same way.
        B, T, D = x.shape
        q = self.WQ(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.WK(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.WV(x).view(B, T, self.h, self.dh).transpose(1, 2)
        sc = (q @ k.transpose(-1, -2)) / np.sqrt(self.dh)
        m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        sc = sc.masked_fill(m, -1e9).softmax(-1)
        z = (sc @ v).transpose(1, 2).reshape(B, T, D)
        return self.WO(z)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        h = self.fc_in(self.n2(x))
        h = F.relu(h) if self.act == "relu" else torch.tanh(h)
        return x + self.fc_out(h)


class LMVariant(nn.Module):
    def __init__(self, variant, d, vocab, n_heads=4, n_layers=2):
        super().__init__()
        mode = "standard" if variant == "standard" else "fixnorm"
        act = "tanh" if variant.endswith("tanh") else "relu"
        self.variant, self.n_layers = variant, n_layers
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, n_heads, mode, act)
                                     for _ in range(n_layers)])
        self.nf = Norm(d, mode)
        self.unembed = nn.Linear(d, vocab, bias=False)

    def embed_stream(self, toks):
        return self.embed(toks) + self.pos[None, :toks.shape[1]]

    def forward(self, toks):
        x = self.embed_stream(toks)
        for b in self.blocks:
            x = b(x)
        return self.unembed(self.nf(x))

    def set_calibrating(self, f):
        for m in self.modules():
            if isinstance(m, Norm):
                m.calibrating = f


def export_W(model, d):
    """Torch weights -> the float64 dict src/bounds.py consumes. This is the
    ONLY thing that crosses the GPU/CPU boundary."""
    W = {"d_model": d, "n_heads": 4, "n_layers": model.n_layers,
         "seq_len": SEQ_LEN, "ln_eps": EPS, "blocks": []}
    for b in model.blocks:
        f = lambda t: t.detach().cpu().numpy().astype(np.float64)
        W["blocks"].append({
            "ln1_g": f(b.n1.g), "ln1_b": f(b.n1.b),
            "ln2_g": f(b.n2.g), "ln2_b": f(b.n2.b),
            "scale1": float(b.n1.scale), "scale2": float(b.n2.scale),
            "WQ": f(b.WQ.weight), "WK": f(b.WK.weight),
            "WV": f(b.WV.weight), "WO": f(b.WO.weight),
            "fc_in_W": f(b.fc_in.weight), "fc_in_b": f(b.fc_in.bias),
            "fc_out_W": f(b.fc_out.weight), "fc_out_b": f(b.fc_out.bias)})
    return W


def save_W(W, path):
    flat = {"__meta__": json.dumps({k: W[k] for k in
                                    ("d_model", "n_heads", "n_layers",
                                     "seq_len", "ln_eps")})}
    for i, b in enumerate(W["blocks"]):
        for k, v in b.items():
            flat[f"b{i}.{k}"] = np.asarray(v, dtype=np.float64)
    np.savez_compressed(path, **flat)


def load_W(path):
    z = np.load(path, allow_pickle=False)
    W = json.loads(str(z["__meta__"]))
    W["blocks"] = []
    for i in range(W["n_layers"]):
        blk = {}
        for key in z.files:
            if key.startswith(f"b{i}."):
                name = key.split(".", 1)[1]
                v = z[key]
                blk[name] = float(v) if name.startswith("scale") else v
        W["blocks"].append(blk)
    return W


# ------------------------------------------------------------------- training
def pick_device(spec):
    if spec != "auto":
        return torch.device(spec)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, X, Y, vocab, chunk=4096):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, X.shape[0], chunk):
        x, y = X[i:i + chunk], Y[i:i + chunk]
        lg = model(x)
        ce = F.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1),
                             reduction="sum")
        tot += float(ce); n += y.numel()
    model.train()
    return tot / n


def train_one(variant, seed, d, L, C, A, dev, VX, VY, TX, TY):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LMVariant(variant, d, C["vocab"], n_layers=L).to(dev)
    rng = np.random.default_rng(seed)

    x0, _ = sample(A.batch, rng, C["train"])
    model.set_calibrating(True)
    with torch.no_grad():
        model(torch.from_numpy(x0).to(dev))
    model.set_calibrating(False)

    opt = torch.optim.Adam(model.parameters(), lr=A.lr)
    # Depth needs a warmup that L=2 did not. Without it the L=12 cells diverge
    # at lr=3e-3 and the perplexity comparison would measure an optimizer
    # failure rather than the capacity cost of a fixed normalizer.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, A.warmup)))

    curve, diverged = [], False
    for step in range(A.steps):
        x, y = sample(A.batch, rng, C["train"])
        logits = model(torch.from_numpy(x).to(dev))
        loss = F.cross_entropy(logits.reshape(-1, C["vocab"]),
                               torch.from_numpy(y).to(dev).reshape(-1))
        opt.zero_grad(); loss.backward()
        gn = float(torch.sqrt(sum((p.grad ** 2).sum()
                                  for p in model.parameters()
                                  if p.grad is not None)))
        if A.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), A.clip)
        opt.step(); sched.step()
        if not np.isfinite(float(loss.detach())):
            diverged = True
            break
        if step % A.eval_every == 0 or step == A.steps - 1:
            vce = evaluate(model, TX, TY, C["vocab"])
            curve.append({"step": step, "train_loss": float(loss.detach()),
                          "grad_norm": gn, "val_ce": vce,
                          "val_ppl": float(np.exp(vce))})

    vce = evaluate(model, VX, VY, C["vocab"])
    return model, {
        "variant": variant, "seed": seed, "d_model": d, "n_layers": L,
        "curve": curve, "val_ce": vce, "val_ppl": float(np.exp(vce)),
        "val_bpc": vce / np.log(2.0), "diverged": diverged,
        "final_grad_norm": curve[-1]["grad_norm"] if curve else float("nan"),
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }


# --------------------------------------------------------------- verification
def gap_on_weights(W, variant, d, rho, C, seed=0):
    """Post-training relaxation gap on the UNCHANGED NumPy engine.

    Same construction as c23.gap_on_trained: 4 unit directions on the final
    residual position, radius rho, nominal point from a real held-out window.
    Overflow is caught and reported rather than allowed to poison a mean --
    at L>=8 the standard variant genuinely overflows float64.
    """
    from src import bounds as bd
    import c18_variants as c18

    rng = np.random.default_rng(seed)
    x, _ = sample(1, rng, C["val"])

    emb = W.pop("__emb__")
    dirs = rng.normal(size=(4, d))
    U = np.zeros((4, SEQ_LEN, d))
    U[np.arange(4), SEQ_LEN - 1, :] = dirs / np.linalg.norm(dirs, axis=1,
                                                            keepdims=True)
    z = bd.HZono(emb[None], np.full((1, 4), rho)[:, :, None, None] * U[None])

    unstable = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for l in range(W["n_layers"]):
            z, st = c18.block(z, W["blocks"][l], W["n_heads"], variant, EPS)
            unstable += st["unstable"]
        bound = float(z.width().max())

        al = rng.uniform(-rho, rho, size=(2000, 4))
        xs = np.einsum("bk,ktd->btd", al, U) + emb[None]
        out = c18.exact_forward(W, variant, xs)
        att = float((out.max(0) - out.min(0)).max())
        lo, hi = z.bounds()
        viol = float(np.max(np.maximum(lo[0][None] - out, out - hi[0][None])))

    gap = bound / max(att, 1e-300) if np.isfinite(bound) else float("inf")
    return {"bound_width": bound, "attained_width": att,
            "relaxation_gap": gap,
            "gap_finite": bool(np.isfinite(gap)),
            "gap_meaningful": bool(np.isfinite(gap) and gap <= GAP_MEANINGFUL),
            "unstable_relus": int(unstable),
            "max_containment_violation": viol,
            "prompt": "".join(C["chars"][i] for i in x[0])}


# --------------------------------------------------------------------- driver
def fingerprint(A, C):
    """Everything that changes a cell's numbers. Training and verification are
    fingerprinted separately: a verification-only rerun must not invalidate
    hours of GPU training just because --rho moved."""
    key = {
        "steps": A.steps, "batch": A.batch, "lr": A.lr, "warmup": A.warmup,
        "seq_len": SEQ_LEN, "val_seqs": A.val_seqs,
        "eval_every": A.eval_every, "data": C["sha256"],
    }
    # Only perturb the hash when clipping is actually on, so the cells already
    # computed before --clip existed keep their fingerprint and stay valid.
    if A.clip > 0:
        key["clip"] = A.clip
    train = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    ver = hashlib.sha256(json.dumps(
        {"rho": A.rho, "train": train}, sort_keys=True).encode()).hexdigest()[:16]
    return train, ver


def run_cell(v, s, d, L, A, C, dev, VX, VY, TX, TY, fp_t, fp_v, stamp, verify,
             save_weights=True):
    """One (variant, seed, width, depth) cell. Training and verification cache
    independently so a reclaimed Colab VM costs at most one cell.

    save_weights is False outside the verification subgrid. float64 weights for
    L=12/d=512 are ~300 MB per cell and Colab's /content is wiped when a session
    is reclaimed, so every cell we intend to verify has to be pulled back to
    local disk. Cells we only need perplexity from ship a few KB of JSON."""
    os.makedirs(A.cache, exist_ok=True)
    tag = f"L{L}_d{d}_{v}_s{s}"
    pj = os.path.join(A.cache, tag + ".json")
    pw = os.path.join(A.cache, tag + ".npz")

    rec = None
    if os.path.exists(pj):
        try:
            cand = json.load(open(pj))
            if cand.get("fp_train") == fp_t:
                rec = cand
        except (ValueError, KeyError):
            pass

    if rec is None:
        t = time.time()
        model, r = train_one(v, s, d, L, C, A, dev, VX, VY, TX, TY)
        r["train_seconds"] = time.time() - t
        if save_weights:
            W = export_W(model, d)
            # The nominal point for verification is drawn with the SAME rng
            # sequence the gap pass uses, so an offline verify reproduces an
            # in-process one exactly.
            rng = np.random.default_rng(s)
            xg, _ = sample(1, rng, C["val"])
            with torch.no_grad():
                emb = (model.embed_stream(torch.from_numpy(xg).to(dev))
                       .cpu().numpy().astype(np.float64)[0])
            save_W(W, pw)
            np.savez_compressed(pw.replace(".npz", "_emb.npz"), emb=emb)
        r["weights_saved"] = bool(save_weights)
        r["device"] = dev.type
        rec = {"fp_train": fp_t, "run": r, "device": dev.type}
        json.dump(rec, open(pj, "w"), indent=2)
        stamp(f"  L={L:2d} d={d:3d} {v:8s} s={s} ppl={r['val_ppl']:.4f} "
              f"bpc={r['val_bpc']:.4f} div={r['diverged']} "
              f"[{r['train_seconds']:.0f}s train]")
    else:
        # CUDA and CPU do not produce identical weights after 3000 Adam steps,
        # and the fingerprint deliberately does not include device (adding it
        # would invalidate every banked cell). Measured drift at L=4/d=128
        # fixnorm: mean ppl 6.4245 on T4 vs 6.3851 on CPU, i.e. ~0.04 -- the
        # same order as the penalties being reported. So a cell trained on a
        # different device than its opposite variant is NOT a valid pairing.
        cached_dev = rec.get("device", "unknown")
        warn = "" if cached_dev == dev.type else f"  !! DEVICE MISMATCH ({cached_dev})"
        stamp(f"  L={L:2d} d={d:3d} {v:8s} s={s} CACHED-TRAIN "
              f"ppl={rec['run']['val_ppl']:.4f}{warn}")

    if verify and rec.get("fp_verify") != fp_v:
        if not os.path.exists(pw):
            stamp(f"  L={L:2d} d={d:3d} {v:8s} s={s} no weights, skip verify")
            return rec["run"]
        t = time.time()
        W = load_W(pw)
        W["__emb__"] = np.load(pw.replace(".npz", "_emb.npz"))["emb"]
        g = gap_on_weights(W, v, d, A.rho, C, seed=s)
        g["gap_seconds"] = time.time() - t
        rec["run"]["trained_gap"] = g
        rec["fp_verify"] = fp_v
        json.dump(rec, open(pj, "w"), indent=2)
        gs = f"{g['relaxation_gap']:.4g}" if g["gap_finite"] else "OVERFLOW"
        stamp(f"  L={L:2d} d={d:3d} {v:8s} s={s} gap={gs} "
              f"meaningful={g['gap_meaningful']} "
              f"unstable={g['unstable_relus']} viol={g['max_containment_violation']:+.1e} "
              f"[{g['gap_seconds']:.1f}s]")
    return rec["run"]


def selfcheck(A, C, dev, VX, VY, TX, TY, stamp):
    """c24 replicates c23's corpus handling by hand. Prove it by reproducing
    c23's published cells; a mismatch means the replication drifted."""
    ref_path = os.path.join(ROOT, "results", "c23.json")
    if not os.path.exists(ref_path):
        stamp("selfcheck: results/c23.json absent, skipping")
        return True
    ref = json.load(open(ref_path))
    ok = True
    for v in ("standard", "fixnorm"):
        for s in (0, 1):
            exp = [r for r in ref["runs"] if r["d_model"] == 64
                   and r["variant"] == v and r["seed"] == s]
            if not exp:
                continue
            _, r = train_one(v, s, 64, 2, C, A, dev, VX, VY, TX, TY)
            got, want = r["val_ppl"], exp[0]["val_ppl"]
            rel = abs(got - want) / want
            flag = "OK" if rel < 2e-2 else "MISMATCH"
            if rel >= 2e-2:
                ok = False
            stamp(f"selfcheck L=2 d=64 {v:8s} s={s} c24={got:.4f} "
                  f"c23={want:.4f} rel={rel:.2e} {flag}")
    return ok


def main(argv=None):
    A = build_args(argv)
    torch.set_num_threads(A.threads)
    t0 = time.time()

    def stamp(m):
        print(f"[{time.time() - t0:8.1f}s] {m}", flush=True)

    C = load_corpus()
    dev = torch.device("cpu") if A.verify_only else pick_device(A.device)

    _vrng = np.random.default_rng(20260801)
    VAL_X, VAL_Y = sample(A.val_seqs, _vrng, C["val"])
    VX, VY = torch.from_numpy(VAL_X).to(dev), torch.from_numpy(VAL_Y).to(dev)
    ntraj = max(1, A.val_seqs // 4)
    TX, TY = VX[:ntraj], VY[:ntraj]

    fp_t, fp_v = fingerprint(A, C)
    gpu = (torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu")
    stamp(f"device={dev.type} [{gpu}] fp_train={fp_t} fp_verify={fp_v} "
          f"vocab={C['vocab']} val_tokens={VAL_Y.size}")

    if A.selfcheck:
        ok = selfcheck(A, C, dev, VX, VY, TX, TY, stamp)
        stamp(f"selfcheck {'PASSED' if ok else 'FAILED'}")
        if not ok:
            return 1

    widths = [int(x) for x in A.widths.split(",")]
    layers = [int(x) for x in A.layers.split(",")]
    variants = A.variants.split(",")
    vw = {int(x) for x in A.verify_widths.split(",")}
    vl = {int(x) for x in A.verify_layers.split(",")}

    runs = []
    for L in layers:
        for d in widths:
            in_subgrid = (L in vl) and (d in vw)
            do_verify = (not A.train_only) and in_subgrid
            for v in variants:
                for s in range(A.seeds):
                    runs.append(run_cell(v, s, d, L, A, C, dev, VX, VY, TX, TY,
                                         fp_t, fp_v, stamp, do_verify,
                                         save_weights=in_subgrid))

    # ------------------------------------------------------------- summarise
    summary = {}
    for L in layers:
        for d in widths:
            cell = {}
            for v in variants:
                rs = [r for r in runs if r["d_model"] == d
                      and r["n_layers"] == L and r["variant"] == v]
                if not rs:
                    continue
                ppl = np.array([r["val_ppl"] for r in rs])
                fin = [r for r in rs if r.get("trained_gap")]
                good = [r["trained_gap"]["relaxation_gap"] for r in fin
                        if r["trained_gap"]["gap_finite"]]
                se = float(ppl.std(ddof=1) / np.sqrt(len(ppl))) if len(ppl) > 1 else 0.0
                cell[v] = {
                    "n_seeds": len(ppl),
                    "val_ppl_mean": float(ppl.mean()),
                    "val_ppl_std": float(ppl.std(ddof=1)) if len(ppl) > 1 else 0.0,
                    "val_ppl_sem": se,
                    "any_diverged": bool(any(r["diverged"] for r in rs)),
                    "n_params": rs[0]["n_params"],
                    "verified": len(fin),
                    "gap_median": float(np.median(good)) if good else None,
                    "gap_all_finite": bool(fin) and len(good) == len(fin),
                    "gap_meaningful": bool(good) and float(np.median(good)) <= GAP_MEANINGFUL,
                    "unstable_median": (float(np.median(
                        [r["trained_gap"]["unstable_relus"] for r in fin]))
                        if fin else None),
                    "max_containment_violation": (float(np.max(
                        [r["trained_gap"]["max_containment_violation"] for r in fin]))
                        if fin else None),
                }
            if "standard" in cell and "fixnorm" in cell:
                b, f = cell["standard"], cell["fixnorm"]
                dp = f["val_ppl_mean"] - b["val_ppl_mean"]
                sed = float(np.sqrt(b["val_ppl_sem"] ** 2 + f["val_ppl_sem"] ** 2))
                cell["ppl_penalty"] = dp
                cell["ppl_penalty_ci95"] = [dp - 1.96 * sed, dp + 1.96 * sed]
                cell["ppl_penalty_significant"] = bool((dp - 1.96 * sed) > 0)
                cell["ppl_penalty_pct"] = 100.0 * dp / b["val_ppl_mean"]
                # A gain ratio is only reported when BOTH arms are inside the
                # meaningful band. Above it the ratio is arithmetic on two
                # numbers that no longer certify anything.
                if b["gap_meaningful"] and f["gap_meaningful"]:
                    cell["gap_gain"] = b["gap_median"] / f["gap_median"]
                    cell["gap_gain_reportable"] = True
                else:
                    cell["gap_gain"] = None
                    cell["gap_gain_reportable"] = False
                    cell["gap_gain_withheld_reason"] = (
                        "at least one arm exceeds GAP_MEANINGFUL="
                        f"{GAP_MEANINGFUL:g}; the bound no longer certifies")
            summary[f"L{L}_d{d}"] = cell

    rep = {
        "config": {"steps": A.steps, "batch": A.batch, "lr": A.lr,
                   "warmup": A.warmup, "seeds": A.seeds, "widths": widths,
                   "layers": layers, "variants": variants, "seq_len": SEQ_LEN,
                   "vocab": C["vocab"], "rho": A.rho, "n_heads": 4,
                   "device": dev.type, "gpu": gpu,
                   "gap_meaningful_threshold": GAP_MEANINGFUL,
                   "verify_layers": sorted(vl), "verify_widths": sorted(vw),
                   "fp_train": fp_t, "fp_verify": fp_v,
                   "wall_seconds_total": round(time.time() - t0, 1)},
        "data": {"name": "tinyshakespeare", "sha256": C["sha256"],
                 "chars": C["n_chars"], "vocab": C["vocab"],
                 "split": "contiguous 90/10 by position"},
        "runs": runs, "summary": summary,
        "note": ("Split-metric sweep. Perplexity is measured on the full L x d "
                 "grid; the relaxation gap is measured only on the subgrid "
                 "where it still certifies, and gain ratios are withheld "
                 "wherever either arm exceeds the meaningfulness threshold. "
                 "Training may run on CUDA; verification always runs on the "
                 "unmodified NumPy engine in src/bounds.py, on CPU, in "
                 "float64."),
    }
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    json.dump(rep, open(A.out, "w"), indent=2, default=str)
    stamp(f"wrote {A.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
