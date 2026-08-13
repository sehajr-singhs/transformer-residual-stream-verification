"""c20 -- expressivity boundaries under a non-saturated task.

c19 found no accuracy penalty for the fixed-norm variants, but every variant hit
100% on the routing task. A task that everything solves perfectly has no power
to discriminate trainability, so that result bounds nothing. This runs the same
comparison on a task hard enough that the models do not saturate.

The hard task (defined HERE, not in src/task.py, so the original task and
results/baseline.json stay reproducible):

    positions 0..5   six content slots holding feature tokens 0..N_FEAT-1
    position 6       a distractor token, never used by the label rule
    position 7       a selector token pointing at slot i

    v = slots[i]
    if v is a DISTRACTOR feature (>= N_REAL):  target = slots[0]
    else:                                      target = slots[v % N_SLOT]

That is genuinely two-hop -- the value read by the first hop is the index for
the second -- plus a non-linear branch on whether the first hop lands on a
distractor. Two attention layers can express it, one cannot, so a 2-layer
d_model=32 model sits near its capacity rather than far above it.

Reported: accuracy, cross-entropy trajectory, gradient norms, and the
post-training relaxation gap, for standard / fixnorm / fixnorm_tanh under
identical seeds and batch streams.
"""
import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from src import bounds as bd
import c18_variants as c18
import c19_trainability as c19

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--d-model", type=int, default=32)
ap.add_argument("--variants", default="standard,fixnorm,fixnorm_tanh")
ap.add_argument("--n-feat", type=int, default=12)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "c20.json"))
class _A: pass
args = _A()
for _a in ap._actions:
    if _a.dest != 'help':
        setattr(args, _a.dest, _a.default)

def _cli():
    global args
    args = ap.parse_args()

t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

# ------------------------------------------------------------------- the task
N_FEAT = args.n_feat
N_REAL = 8           # 0..7 are real; 8..11 are the four distractor features
N_SLOT = 6           # content slots
SEQ_LEN = 8          # 6 slots + 1 distractor position + 1 selector
VOCAB = N_FEAT + N_SLOT
UNSAFE = (6, 7)      # unsafe features, both in the REAL range so they can be targets
EPS = 1e-5


# Fraction of examples that use the two-hop rule; the rest are one-hop
# (target = slots[sel]) and much easier. This is the difficulty knob c22
# calibrates with. n_feat is a terrible knob for that: measured accuracy runs
# 0.952 at n_feat=8 and 0.214 at n_feat=70, a cliff with essentially no
# resolution inside an 0.80-0.90 target band. TWO_HOP_P is smooth and monotone
# and leaves the class count fixed, so accuracies stay comparable.
# Default 1.0 reproduces the c20/c21 task exactly.
TWO_HOP_P = 1.0


def sample(n, rng, safe_only=False):
    for _ in range(64):
        slots = rng.integers(0, N_FEAT, size=(n, N_SLOT))
        distract = rng.integers(0, N_FEAT, size=(n, 1))
        sel = rng.integers(0, N_SLOT, size=n)
        v = slots[np.arange(n), sel]
        idx2 = np.where(v >= N_REAL, 0, v % N_SLOT)
        two_hop = rng.random(n) < TWO_HOP_P
        idx2 = np.where(two_hop, idx2, sel)
        tgt = slots[np.arange(n), idx2]
        toks = np.concatenate([slots, distract, (N_FEAT + sel)[:, None]], axis=1)
        if not safe_only:
            return toks.astype(np.int64), tgt.astype(np.int64)
        keep = ~np.isin(tgt, UNSAFE)
        # max(1, ...) matters: with n=1 the old guard `>= n//2` was `>= 0`,
        # which is always true and could return an EMPTY batch.
        if keep.sum() >= max(1, n // 2):
            return toks[keep].astype(np.int64), tgt[keep].astype(np.int64)
    return toks.astype(np.int64), tgt.astype(np.int64)


def unsafe_margin(logits):
    um = np.max(logits[..., list(UNSAFE)], axis=-1)
    m = np.ones(logits.shape[-1], dtype=bool)
    for u in UNSAFE:
        m[u] = False
    return um - np.max(logits[..., m], axis=-1)


class HardVariant(nn.Module):
    """c19's blocks, retargeted at this task's vocabulary and length."""

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


def train_one(variant, seed, d):
    torch.manual_seed(seed); np.random.seed(seed)
    model = HardVariant(variant, d)
    rng = np.random.default_rng(seed)
    tk, _ = sample(args.batch, rng)
    model.set_calibrating(True)
    with torch.no_grad():
        model(torch.from_numpy(tk))
    model.set_calibrating(False)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    curve = []
    for step in range(args.steps):
        tk, tg = sample(args.batch, rng)
        logits = model(torch.from_numpy(tk))[:, -1, :]
        loss = F.cross_entropy(logits, torch.from_numpy(tg))
        opt.zero_grad(); loss.backward()
        gn = float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                                  if p.grad is not None)))
        opt.step()
        if step % 200 == 0 or step == args.steps - 1:
            curve.append({"step": step, "loss": float(loss), "grad_norm": gn})
    model.eval()
    ev = np.random.default_rng(seed + 9999)
    tk, tg = sample(8192, ev)
    with torch.no_grad():
        lg = model(torch.from_numpy(tk))[:, -1, :]
    acc = float((lg.argmax(-1).numpy() == tg).mean())
    ce = float(F.cross_entropy(lg, torch.from_numpy(tg)))
    tk_s, _ = sample(4096, ev, safe_only=True)
    with torch.no_grad():
        lg_s = model(torch.from_numpy(tk_s))[:, -1, :].numpy()
    mg = unsafe_margin(lg_s)
    return model, {"variant": variant, "seed": seed, "curve": curve,
                   "test_accuracy": acc, "test_ce": ce,
                   "final_grad_norm": curve[-1]["grad_norm"],
                   "frac_margin_negative": float((mg < 0).mean()),
                   "unsafe_margin_mean": float(mg.mean())}


def gap_on_trained(model, variant, d, rho=1e-4, seed=0):
    """Post-training relaxation gap: zonotope bound width / attained width."""
    W = export_W(model, d)
    rng = np.random.default_rng(seed)
    # draw a batch and take one row: a size-1 safe_only draw can come back empty
    tk, _ = sample(64, rng, safe_only=True)
    tk = tk[:1]
    with torch.no_grad():
        emb = model.embed_stream(torch.from_numpy(tk)).numpy().astype(np.float64)[0]
    dirs = rng.normal(size=(4, d))
    U = np.zeros((4, SEQ_LEN, d))
    U[np.arange(4), SEQ_LEN - 1, :] = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    z = bd.HZono(emb[None], np.full((1, 4), rho)[:, :, None, None] * U[None])
    in_w = float(z.width().max())
    unstable = 0
    for l in range(W["n_layers"]):
        z, st = c18.block(z, W["blocks"][l], W["n_heads"], variant, EPS)
        unstable += st["unstable"]
    bound = float(z.width().max())
    al = rng.uniform(-rho, rho, size=(2000, 4))
    xs = np.einsum("bk,ktd->btd", al, U) + emb[None]
    out = c18.exact_forward(W, variant, xs)
    att = float((out.max(0) - out.min(0)).max())
    viol = 0.0
    lo, hi = z.bounds()
    viol = float(np.max(np.maximum(lo[0][None] - out, out - hi[0][None])))
    return {"bound_width": bound, "attained_width": att,
            "relaxation_gap": bound / max(att, 1e-300),
            "unstable_relus": unstable, "max_containment_violation": viol}


def main():
    _cli()
    VARIANTS = args.variants.split(",")
    runs = []
    for v in VARIANTS:
        for s in range(args.seeds):
            model, r = train_one(v, s, args.d_model)
            r["trained_gap"] = gap_on_trained(model, v, args.d_model, seed=s)
            runs.append(r)
            stamp(f"  {v:13s} seed={s} acc={r['test_accuracy']:.4f} "
                  f"ce={r['test_ce']:.4f} gnorm={r['final_grad_norm']:.3e} "
                  f"gap={r['trained_gap']['relaxation_gap']:.3e} "
                  f"viol={r['trained_gap']['max_containment_violation']:+.1e}")

    summary = {}
    for v in VARIANTS:
        rs = [r for r in runs if r["variant"] == v]
        accs = [r["test_accuracy"] for r in rs]
        summary[v] = {
            "test_accuracy_mean": float(np.mean(accs)),
            "test_accuracy_std": float(np.std(accs)),
            "test_ce_mean": float(np.mean([r["test_ce"] for r in rs])),
            "final_grad_norm_mean": float(np.mean([r["final_grad_norm"] for r in rs])),
            "trained_relaxation_gap_median": float(np.median(
                [r["trained_gap"]["relaxation_gap"] for r in rs])),
            "frac_margin_negative_mean": float(np.mean(
                [r["frac_margin_negative"] for r in rs])),
            "max_containment_violation": float(np.max(
                [r["trained_gap"]["max_containment_violation"] for r in rs])),
        }
    base_acc = summary["standard"]["test_accuracy_mean"]
    base_gap = summary["standard"]["trained_relaxation_gap_median"]
    for v in VARIANTS:
        summary[v]["accuracy_delta_vs_standard"] = \
            summary[v]["test_accuracy_mean"] - base_acc
        summary[v]["gap_improvement_vs_standard"] = \
            base_gap / max(summary[v]["trained_relaxation_gap_median"], 1e-300)

    sat = base_acc
    rep = {"config": {"steps": args.steps, "batch": args.batch, "lr": args.lr,
                      "seeds": args.seeds, "d_model": args.d_model,
                      "task": {"N_FEAT": N_FEAT, "N_REAL": N_REAL,
                               "N_SLOT": N_SLOT, "SEQ_LEN": SEQ_LEN,
                               "VOCAB": VOCAB, "hops": 2,
                               "distractor_features": N_FEAT - N_REAL}},
           "runs": runs, "summary": summary,
           "non_saturated": bool(0.70 <= sat <= 0.95),
           "baseline_accuracy": sat,
           "soundness": {"max_containment_violation": max(
               summary[v]["max_containment_violation"] for v in VARIANTS)},
           "note": ("Task defined inside this module so src/task.py and "
                    "results/baseline.json remain untouched. The CROWN-side gap "
                    "was NOT computed: audit/crown_reference.py implements "
                    "LayerNorm and ReLU only, so extending it to the fixnorm and "
                    "tanh variants would be new unvalidated code. Only the hybrid "
                    "zonotope gap is reported.")}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)
    stamp(f"wrote {args.out}  (baseline acc {sat:.4f}, "
          f"non-saturated={rep['non_saturated']})")
    print(json.dumps(summary, indent=2))



if __name__ == "__main__":
    main()
