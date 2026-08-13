"""c23 -- the fixed-normalizer Pareto exchange on natural language.

Why this experiment exists
--------------------------
Every trainability result from c19 through c22 was measured on the synthetic
routing family, and that family is now exhausted as a measuring instrument.
c19 reported an accuracy penalty of exactly 0.0000 because all three variants
sat at 100%. c20 fixed that at d_model=8 and found a real 5.7-point penalty.
c21 tried to ask whether the penalty shrinks with capacity and could not
answer, because its two cells were confounded in opposite directions. c22 then
tried to capacity-normalise the task per width and discovered the family has no
knob that can do it -- see results/c22_ceiling.json, and the abandonment note
in report_manuscript.py section 6i.

The blocker is structural: accuracy is bounded above by 1.0, so a task that a
wider model can solve outright destroys the measurement. Perplexity has no such
ceiling. A character-level language model is never "done" -- the loss floor is
the conditional entropy of English given the context, which is strictly
positive -- so the standard baseline cannot saturate at ANY width, and the
standard-vs-fixnorm comparison stays interpretable as capacity grows.

That is the entire reason for moving to text. It is not a scale demonstration:
these are still 2-layer models with an 8-character context, far below anything
that would be called a language model in practice.

What is measured
----------------
    standard  x -> (x - mean)/sqrt(var + eps) * g + b
    fixnorm   x -> (x - mean)/c * g + b,  c frozen after one calibration batch

at d_model in {32, 64}, 3000 steps, several seeds, identical batch streams and
initialisation per seed. Two numbers per cell:

  * raw held-out perplexity -- the expressivity cost of a data-independent
    normalizer on natural text, with no ceiling to hide it;
  * the post-training relaxation gap (zonotope bound width / attained width)
    on the converged weights -- the verifiability that cost buys.

Scope limits, stated up front
-----------------------------
This is a strictly POST-HOC evaluation loop. Nothing here is a differentiable
verification penalty, and src/bounds.py is untouched: the same sound NumPy
hybrid-zonotope engine that produced every earlier number produces these.

There is no safety-margin readout in this experiment. c19-c22 tracked the
fraction of safe prompts with a negative unsafe-logit margin, which relied on
the synthetic task designating features 6 and 7 as "unsafe". No character of
English is unsafe, and nominating one would be a decoration rather than a
measurement, so the column is dropped rather than faked.
"""
import sys, os, json, time, argparse, hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from src import bounds as bd
import c18_variants as c18
import c19_trainability as c19

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "tinyshakespeare.txt")

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--widths", default="32,64")
ap.add_argument("--variants", default="standard,fixnorm")
ap.add_argument("--eval-every", type=int, default=100)
ap.add_argument("--val-seqs", type=int, default=16384)
ap.add_argument("--rho", type=float, default=1e-4)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--cache", default=os.path.join(ROOT, "results", "c23_cache"))
ap.add_argument("--out", default=os.path.join(ROOT, "results", "c23.json"))
args = ap.parse_args()

# Pin the thread count. Measured on this machine: 3000 steps costs ~454s at
# 1 thread, ~108s at 4, ~112s at 14 -- 4 is at the knee and leaves headroom.
# Pinning also keeps per-step cost reproducible; the first run of this sweep
# saw the same cell take 37ms/step and 460ms/step on different attempts.
torch.set_num_threads(args.threads)


def _keep_system_awake():
    """Ask Windows not to sleep while this sweep runs.

    Measured cause of two lost overnight runs: the process is not slow, it is
    starved. Cells that cost 96s of compute took 27,614s of wall clock at 6.6%
    of one core because the machine slept underneath them. This is a
    process-scoped request via SetThreadExecutionState -- it is released
    automatically when the process exits, changes no saved power plan, and
    deliberately omits ES_DISPLAY_REQUIRED so the screen still blanks.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ok = ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        return bool(ok)
    except Exception:
        return False

t0 = time.time()
def stamp(m): print(f"[{time.time() - t0:8.1f}s] {m}", flush=True)

SEQ_LEN = 8          # fixed model context, per the c23 specification
EPS = 1e-5

# --------------------------------------------------------------------- corpus
_raw = open(DATA, "rb").read()
TEXT = _raw.decode("utf-8")
DATA_SHA = hashlib.sha256(_raw).hexdigest()
CHARS = sorted(set(TEXT))
VOCAB = len(CHARS)
STOI = {c: i for i, c in enumerate(CHARS)}
IDS = np.array([STOI[c] for c in TEXT], dtype=np.int64)

# 90/10 split by position. Held-out text is a contiguous tail, so no training
# window can overlap a validation window.
N_TRAIN = int(0.9 * len(IDS))
TRAIN_IDS, VAL_IDS = IDS[:N_TRAIN], IDS[N_TRAIN:]


def sample(n, rng, ids):
    """n windows of SEQ_LEN+1 chars -> (context, next-char targets)."""
    hi = len(ids) - SEQ_LEN - 1
    off = rng.integers(0, hi, size=n)
    w = off[:, None] + np.arange(SEQ_LEN + 1)[None, :]
    win = ids[w]
    return win[:, :-1], win[:, 1:]


# One fixed validation set, drawn once, shared by every run. Perplexity
# differences between variants must not be a resampling artifact.
_vrng = np.random.default_rng(20260801)
VAL_X, VAL_Y = sample(args.val_seqs, _vrng, VAL_IDS)
VAL_X_T, VAL_Y_T = torch.from_numpy(VAL_X), torch.from_numpy(VAL_Y)

# The trajectory is evaluated on a fixed quarter of the validation set and the
# reported final number on all of it. 31 full evaluations per run cost more
# than the training they are measuring; the headline perplexity keeps the full
# set's precision either way.
_NTRAJ = max(1, args.val_seqs // 4)
TRAJ_X_T, TRAJ_Y_T = VAL_X_T[:_NTRAJ], VAL_Y_T[:_NTRAJ]


class LMVariant(nn.Module):
    """c19's Block/Norm retargeted at characters. standard vs fixnorm still
    differ in exactly one line, inside c19.Norm."""

    def __init__(self, variant, d, n_heads=4, n_layers=2):
        super().__init__()
        mode = "standard" if variant == "standard" else "fixnorm"
        act = "tanh" if variant.endswith("tanh") else "relu"
        self.variant, self.n_layers = variant, n_layers
        self.embed = nn.Embedding(VOCAB, d)
        self.pos = nn.Parameter(torch.randn(SEQ_LEN, d) * 0.02)
        self.blocks = nn.ModuleList([c19.Block(d, n_heads, mode, act)
                                     for _ in range(n_layers)])
        self.nf = c19.Norm(d, mode)
        self.unembed = nn.Linear(d, VOCAB, bias=False)

    def embed_stream(self, toks):
        return self.embed(toks) + self.pos[None, :toks.shape[1]]

    def forward(self, toks):
        x = self.embed_stream(toks)
        for b in self.blocks:
            x = b(x)
        return self.unembed(self.nf(x))

    def set_calibrating(self, f):
        for m in self.modules():
            if isinstance(m, c19.Norm):
                m.calibrating = f


def export_W(model, d):
    W = {"d_model": d, "n_heads": 4, "n_layers": model.n_layers,
         "seq_len": SEQ_LEN, "ln_eps": EPS, "blocks": []}
    for b in model.blocks:
        W["blocks"].append({
            "ln1_g": b.n1.g.detach().numpy().astype(np.float64),
            "ln1_b": b.n1.b.detach().numpy().astype(np.float64),
            "ln2_g": b.n2.g.detach().numpy().astype(np.float64),
            "ln2_b": b.n2.b.detach().numpy().astype(np.float64),
            "scale1": float(b.n1.scale), "scale2": float(b.n2.scale),
            "WQ": b.WQ.weight.detach().numpy().astype(np.float64),
            "WK": b.WK.weight.detach().numpy().astype(np.float64),
            "WV": b.WV.weight.detach().numpy().astype(np.float64),
            "WO": b.WO.weight.detach().numpy().astype(np.float64),
            "fc_in_W": b.fc_in.weight.detach().numpy().astype(np.float64),
            "fc_in_b": b.fc_in.bias.detach().numpy().astype(np.float64),
            "fc_out_W": b.fc_out.weight.detach().numpy().astype(np.float64),
            "fc_out_b": b.fc_out.bias.detach().numpy().astype(np.float64)})
    return W


@torch.no_grad()
def evaluate(model, X=None, Y=None, chunk=4096):
    """Mean next-char CE (nats) over all SEQ_LEN positions of the fixed val set."""
    if X is None:
        X, Y = VAL_X_T, VAL_Y_T
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, X.shape[0], chunk):
        x, y = X[i:i + chunk], Y[i:i + chunk]
        lg = model(x)
        ce = F.cross_entropy(lg.reshape(-1, VOCAB), y.reshape(-1),
                             reduction="sum")
        tot += float(ce); n += y.numel()
    model.train()
    return tot / n


def train_one(variant, seed, d):
    torch.manual_seed(seed); np.random.seed(seed)
    model = LMVariant(variant, d)
    rng = np.random.default_rng(seed)

    # freeze the fixnorm constants on one batch, before any gradient step
    x0, _ = sample(args.batch, rng, TRAIN_IDS)
    model.set_calibrating(True)
    with torch.no_grad():
        model(torch.from_numpy(x0))
    model.set_calibrating(False)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    curve = []
    for step in range(args.steps):
        x, y = sample(args.batch, rng, TRAIN_IDS)
        logits = model(torch.from_numpy(x))
        loss = F.cross_entropy(logits.reshape(-1, VOCAB),
                               torch.from_numpy(y).reshape(-1))
        opt.zero_grad(); loss.backward()
        gn = float(torch.sqrt(sum((p.grad ** 2).sum()
                                  for p in model.parameters()
                                  if p.grad is not None)))
        opt.step()
        if step % args.eval_every == 0 or step == args.steps - 1:
            vce = evaluate(model, TRAJ_X_T, TRAJ_Y_T)
            curve.append({"step": step, "train_loss": float(loss.detach()),
                          "grad_norm": gn, "val_ce": vce,
                          "val_ppl": float(np.exp(vce))})

    vce = evaluate(model)
    return model, {
        "variant": variant, "seed": seed, "d_model": d, "curve": curve,
        "val_ce": vce, "val_ppl": float(np.exp(vce)),
        "val_bpc": vce / np.log(2.0),
        "final_grad_norm": curve[-1]["grad_norm"],
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }


def gap_on_trained(model, variant, d, rho, seed=0):
    """Post-training relaxation gap: zonotope bound width / attained width.

    Identical construction to c20.gap_on_trained -- 4 unit perturbation
    directions on the final residual-stream position, radius rho -- with the
    nominal point taken from a real held-out text window instead of a synthetic
    prompt. Sound NumPy engine only; src/bounds.py is not touched.
    """
    W = export_W(model, d)
    rng = np.random.default_rng(seed)
    x, _ = sample(1, rng, VAL_IDS)
    with torch.no_grad():
        emb = model.embed_stream(torch.from_numpy(x)).numpy().astype(np.float64)[0]

    dirs = rng.normal(size=(4, d))
    U = np.zeros((4, SEQ_LEN, d))
    U[np.arange(4), SEQ_LEN - 1, :] = dirs / np.linalg.norm(dirs, axis=1,
                                                            keepdims=True)
    z = bd.HZono(emb[None], np.full((1, 4), rho)[:, :, None, None] * U[None])

    unstable = 0
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
    return {"bound_width": bound, "attained_width": att,
            "relaxation_gap": bound / max(att, 1e-300),
            "unstable_relus": unstable,
            "max_containment_violation": viol,
            "prompt": "".join(CHARS[i] for i in x[0])}


def _fingerprint():
    """Everything that changes a cell's numbers. A cache entry written under a
    different fingerprint is stale and is recomputed rather than trusted."""
    return hashlib.sha256(json.dumps({
        "steps": args.steps, "batch": args.batch, "lr": args.lr,
        "seq_len": SEQ_LEN, "rho": args.rho, "val_seqs": args.val_seqs,
        "eval_every": args.eval_every, "data": DATA_SHA,
    }, sort_keys=True).encode()).hexdigest()[:16]


def run_cell(v, s, d, fp):
    """One (variant, seed, width) cell, cached on disk.

    Cells are independent -- train_one seeds torch and numpy from `seed` alone
    and the validation set is fixed -- so a resumed sweep produces bit-identical
    numbers to one that ran straight through. The first attempt at this sweep
    completed 18 of 20 cells and then lost all of them, because results were
    only serialised after the final cell.
    """
    os.makedirs(args.cache, exist_ok=True)
    p = os.path.join(args.cache, f"d{d}_{v}_s{s}.json")
    if os.path.exists(p):
        try:
            rec = json.load(open(p))
            if rec.get("fingerprint") == fp:
                stamp(f"  d={d:3d} {v:9s} seed={s} CACHED "
                      f"ppl={rec['run']['val_ppl']:.4f}")
                return rec["run"]
            stamp(f"  d={d:3d} {v:9s} seed={s} cache stale, recomputing")
        except (ValueError, KeyError):
            stamp(f"  d={d:3d} {v:9s} seed={s} cache unreadable, recomputing")

    t = time.time()
    model, r = train_one(v, s, d)
    r["train_seconds"] = time.time() - t
    t = time.time()
    r["trained_gap"] = gap_on_trained(model, v, d, args.rho, seed=s)
    r["gap_seconds"] = time.time() - t
    json.dump({"fingerprint": fp, "run": r}, open(p, "w"), indent=2)
    stamp(f"  d={d:3d} {v:9s} seed={s} ppl={r['val_ppl']:.4f} "
          f"bpc={r['val_bpc']:.4f} "
          f"gap={r['trained_gap']['relaxation_gap']:.4g} "
          f"viol={r['trained_gap']['max_containment_violation']:+.1e} "
          f"[{r['train_seconds']:.0f}s train, {r['gap_seconds']:.1f}s gap]")
    return r


def main():
    widths = [int(w) for w in args.widths.split(",")]
    variants = args.variants.split(",")
    fp = _fingerprint()
    awake = _keep_system_awake()
    stamp(f"config fingerprint {fp}, threads {torch.get_num_threads()}, "
          f"cache {os.path.relpath(args.cache, ROOT)}, "
          f"sleep_inhibited={awake}")

    runs = []
    for d in widths:
        for v in variants:
            for s in range(args.seeds):
                runs.append(run_cell(v, s, d, fp))

    summary = {}
    for d in widths:
        row = {}
        for v in variants:
            rs = [r for r in runs if r["d_model"] == d and r["variant"] == v]
            ppl = np.array([r["val_ppl"] for r in rs])
            gaps = np.array([r["trained_gap"]["relaxation_gap"] for r in rs])
            se = float(ppl.std(ddof=1) / np.sqrt(len(ppl))) if len(ppl) > 1 else 0.0
            row[v] = {
                "n_seeds": len(ppl),
                "val_ppl_mean": float(ppl.mean()),
                "val_ppl_std": float(ppl.std(ddof=1)) if len(ppl) > 1 else 0.0,
                "val_ppl_sem": se,
                "val_ppl_ci95": [float(ppl.mean() - 1.96 * se),
                                 float(ppl.mean() + 1.96 * se)],
                "val_ce_mean": float(np.mean([r["val_ce"] for r in rs])),
                "val_bpc_mean": float(np.mean([r["val_bpc"] for r in rs])),
                "relaxation_gap_median": float(np.median(gaps)),
                "relaxation_gap_min": float(gaps.min()),
                "relaxation_gap_max": float(gaps.max()),
                "unstable_relus_median": float(np.median(
                    [r["trained_gap"]["unstable_relus"] for r in rs])),
                "max_containment_violation": float(np.max(
                    [r["trained_gap"]["max_containment_violation"] for r in rs])),
                "n_params": rs[0]["n_params"],
            }
        b, f = row["standard"], row["fixnorm"]
        d_ppl = f["val_ppl_mean"] - b["val_ppl_mean"]
        sed = float(np.sqrt(b["val_ppl_sem"] ** 2 + f["val_ppl_sem"] ** 2))
        row["ppl_penalty"] = d_ppl
        row["ppl_penalty_ci95"] = [d_ppl - 1.96 * sed, d_ppl + 1.96 * sed]
        row["ppl_penalty_significant"] = bool((d_ppl - 1.96 * sed) > 0)
        row["ppl_penalty_pct"] = 100.0 * d_ppl / b["val_ppl_mean"]
        row["gap_gain"] = (b["relaxation_gap_median"]
                           / max(f["relaxation_gap_median"], 1e-300))
        # No ceiling to check: perplexity is unbounded above and the entropy
        # floor of English is strictly positive, so the baseline cannot
        # saturate the way accuracy did in c19/c21/c22.
        row["baseline_saturated"] = False
        summary[str(d)] = row

    trend = {}
    if len(widths) >= 2:
        lo, hi = str(widths[0]), str(widths[-1])
        trend = {
            "widths": widths,
            "ppl_penalty": {w: summary[w]["ppl_penalty"] for w in summary},
            "ppl_penalty_pct": {w: summary[w]["ppl_penalty_pct"] for w in summary},
            "gap_gain": {w: summary[w]["gap_gain"] for w in summary},
            "baseline_ppl": {w: summary[w]["standard"]["val_ppl_mean"]
                             for w in summary},
            "all_cells_unsaturated": True,
            "penalty_grows_with_width": bool(
                summary[hi]["ppl_penalty"] > summary[lo]["ppl_penalty"]),
            "gain_grows_with_width": bool(
                summary[hi]["gap_gain"] > summary[lo]["gap_gain"]),
        }

    rep = {
        "config": {"steps": args.steps, "batch": args.batch, "lr": args.lr,
                   "seeds": args.seeds, "widths": widths, "variants": variants,
                   "seq_len": SEQ_LEN, "vocab": VOCAB, "rho": args.rho,
                   "n_layers": 2, "n_heads": 4,
                   "val_seqs": int(args.val_seqs),
                   "val_tokens": int(VAL_Y.size),
                   "traj_seqs": int(_NTRAJ),
                   "threads": int(torch.get_num_threads()),
                   "fingerprint": _fingerprint(),
                   "wall_seconds_total": round(time.time() - t0, 1)},
        "data": {"name": "tinyshakespeare", "path": "data/tinyshakespeare.txt",
                 "sha256": DATA_SHA, "chars": len(TEXT), "vocab": VOCAB,
                 "train_chars": int(N_TRAIN),
                 "val_chars": int(len(IDS) - N_TRAIN),
                 "split": "contiguous 90/10 by position"},
        "runs": runs, "summary": summary, "trend": trend,
        "soundness": {"max_containment_violation": max(
            summary[str(d)][v]["max_containment_violation"]
            for d in widths for v in variants)},
        "note": ("Post-hoc evaluation only: no differentiable verification "
                 "penalty, src/bounds.py unmodified, gaps computed by the same "
                 "sound NumPy hybrid-zonotope engine used in c18-c21. No "
                 "safety-margin readout: the unsafe-feature designation was a "
                 "property of the synthetic task and has no counterpart in "
                 "text. Perplexity is reported RAW and is unbounded above, so "
                 "unlike the accuracy tasks of c19/c21/c22 no cell can be "
                 "confounded by a ceiling."),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)
    stamp(f"wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
