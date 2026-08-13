"""c19 -- does the verification-friendly variant actually TRAIN?

c18 showed that replacing LayerNorm's data-dependent 1/sqrt(var+eps) with a
fixed affine scale removes ~8 orders of relaxation gap at d_model=256. That was
measured on RANDOMLY INITIALISED weights, which leaves the decisive question
open: LayerNorm's scale is data-dependent for optimization reasons, so a variant
that verifies well and trains badly is not a contribution, it is a worse model.

This trains all three variants on the same task, with identical seeds, batches,
initialisation and hyperparameters, and then re-measures the relaxation gap on
the TRAINED weights. Both halves matter: a verifiability gain that survives
random init but not training would be an artifact.

Variants
--------
standard      x -> (x - mean)/sqrt(var + eps) * g + b, ReLU
fixnorm       x -> (x - mean)/c * g + b with c a frozen constant, ReLU
fixnorm_tanh  as fixnorm, with tanh

The frozen constant is calibrated once, before training, from the observed RMS
of the centred activations at that site on a single batch -- the same
construction c18 used. It is a buffer, not a parameter: it never receives
gradient.
"""
import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from src import task, bounds as bd
import c18_variants as c18

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--d-model", type=int, default=32)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "c19.json"))
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

EPS = 1e-5


class Norm(nn.Module):
    """Shared implementation so `standard` and `fixnorm` differ in ONE line."""

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
    def __init__(self, d, n_heads, mode, act):
        super().__init__()
        self.h, self.dh, self.act = n_heads, d // n_heads, act
        self.n1, self.n2 = Norm(d, mode), Norm(d, mode)
        self.WQ = nn.Linear(d, d, bias=False); self.WK = nn.Linear(d, d, bias=False)
        self.WV = nn.Linear(d, d, bias=False); self.WO = nn.Linear(d, d, bias=False)
        self.fc_in = nn.Linear(d, 4 * d); self.fc_out = nn.Linear(4 * d, d)

    def attn(self, x):
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


class Variant(nn.Module):
    def __init__(self, variant, d=32, n_heads=4, n_layers=2):
        super().__init__()
        mode = "standard" if variant == "standard" else "fixnorm"
        act = "tanh" if variant.endswith("tanh") else "relu"
        self.variant, self.n_layers = variant, n_layers
        self.embed = nn.Embedding(task.VOCAB, d)
        self.pos = nn.Parameter(torch.randn(task.SEQ_LEN, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, n_heads, mode, act)
                                     for _ in range(n_layers)])
        self.nf = Norm(d, mode)
        self.unembed = nn.Linear(d, task.VOCAB, bias=False)

    def stream(self, toks):
        x = self.embed(toks) + self.pos[None, :toks.shape[1]]
        for b in self.blocks:
            x = b(x)
        return x

    def forward(self, toks):
        return self.unembed(self.nf(self.stream(toks)))

    def set_calibrating(self, flag):
        for m in self.modules():
            if isinstance(m, Norm):
                m.calibrating = flag


def train_one(variant, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Variant(variant, d=args.d_model)
    rng = np.random.default_rng(seed)
    # calibrate the frozen scales on one batch, before any gradient step
    toks, _ = task.sample_batch(args.batch, rng)
    model.set_calibrating(True)
    with torch.no_grad():
        model(torch.from_numpy(toks))
    model.set_calibrating(False)
    scales = [float(m.scale) for m in model.modules() if isinstance(m, Norm)]

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    curve = []
    for step in range(args.steps):
        tk, tg = task.sample_batch(args.batch, rng)
        logits = model(torch.from_numpy(tk))[:, -1, :]
        loss = F.cross_entropy(logits, torch.from_numpy(tg))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            curve.append({"step": step, "loss": float(loss)})
    # held-out evaluation
    model.eval()
    ev = np.random.default_rng(seed + 9999)
    tk, tg = task.sample_batch(4096, ev)
    with torch.no_grad():
        lg = model(torch.from_numpy(tk))[:, -1, :]
        acc = float((lg.argmax(-1).numpy() == tg).mean())
        ce = float(F.cross_entropy(lg, torch.from_numpy(tg)))
    tk_s, _ = task.sample_batch(2048, ev, safe_only=True)
    with torch.no_grad():
        lg_s = model(torch.from_numpy(tk_s))[:, -1, :].numpy()
    marg = task.unsafe_margin(lg_s)
    return model, {"variant": variant, "seed": seed, "curve": curve,
                   "final_train_loss": curve[-1]["loss"],
                   "test_accuracy": acc, "test_ce": ce,
                   "unsafe_margin_mean": float(marg.mean()),
                   "unsafe_margin_max": float(marg.max()),
                   "frac_margin_negative": float((marg < 0).mean()),
                   "calibrated_scales": scales}


def export_c18(model):
    """Trained weights in the dict format c18_variants.block consumes."""
    d = args.d_model
    W = {"d_model": d, "n_heads": 4, "n_layers": model.n_layers,
         "seq_len": task.SEQ_LEN, "ln_eps": EPS, "blocks": []}
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


def gap_on_trained(model, variant, rho=1e-4, seed=0):
    """Relaxation gap on the TRAINED weights, same metric as c18."""
    W = export_c18(model)
    d, T = args.d_model, task.SEQ_LEN
    rng = np.random.default_rng(seed)
    tk, _ = task.sample_batch(1, rng, safe_only=True)
    with torch.no_grad():
        x_nom = model.stream(torch.from_numpy(tk))[0].numpy().astype(np.float64)
    # the pre-block stream is what the certificate perturbs
    with torch.no_grad():
        emb = (model.embed(torch.from_numpy(tk))
               + model.pos[None, :T]).numpy().astype(np.float64)[0]
    dirs = rng.normal(size=(4, d))
    U = np.zeros((4, T, d))
    U[np.arange(4), T - 1, :] = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
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
    return {"bound_width": bound, "attained_width": att,
            "relaxation_gap": bound / max(att, 1e-300),
            "amplification": bound / max(in_w, 1e-300),
            "unstable_relus": unstable}


def main():
    _cli()
    VARIANTS = ["standard", "fixnorm", "fixnorm_tanh"]
    runs, gaps = [], []
    for v in VARIANTS:
        for s in range(args.seeds):
            model, r = train_one(v, s)
            g = gap_on_trained(model, v, seed=s)
            r["trained_gap"] = g
            runs.append(r)
            stamp(f"  {v:13s} seed={s} acc={r['test_accuracy']:.4f} "
                  f"ce={r['test_ce']:.4f} loss={r['final_train_loss']:.4f} "
                  f"gap={g['relaxation_gap']:.3e}")

    summary = {}
    for v in VARIANTS:
        rs = [r for r in runs if r["variant"] == v]
        accs = [r["test_accuracy"] for r in rs]
        ces = [r["test_ce"] for r in rs]
        gs = [r["trained_gap"]["relaxation_gap"] for r in rs]
        summary[v] = {
            "test_accuracy_mean": float(np.mean(accs)),
            "test_accuracy_std": float(np.std(accs)),
            "test_accuracy_min": float(np.min(accs)),
            "test_ce_mean": float(np.mean(ces)),
            "trained_relaxation_gap_median": float(np.median(gs)),
            "frac_margin_negative_mean": float(np.mean(
                [r["frac_margin_negative"] for r in rs])),
        }
    base = summary["standard"]["test_accuracy_mean"]
    for v in VARIANTS:
        summary[v]["accuracy_delta_vs_standard"] = \
            summary[v]["test_accuracy_mean"] - base
        summary[v]["gap_improvement_vs_standard"] = (
            summary["standard"]["trained_relaxation_gap_median"]
            / max(summary[v]["trained_relaxation_gap_median"], 1e-300))

    rep = {"config": {"steps": args.steps, "batch": args.batch, "lr": args.lr,
                      "seeds": args.seeds, "d_model": args.d_model,
                      "variants": VARIANTS},
           "runs": runs, "summary": summary,
           "note": ("Identical seeds, batch stream, optimizer and step count across "
                    "variants. The frozen scale is calibrated once before training "
                    "and never receives gradient. Relaxation gap is re-measured on "
                    "the TRAINED weights, so a verifiability gain that only "
                    "survives random initialisation would show up here as gone.")}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)
    stamp(f"wrote {args.out}")
    print(json.dumps(summary, indent=2))



if __name__ == "__main__":
    main()
