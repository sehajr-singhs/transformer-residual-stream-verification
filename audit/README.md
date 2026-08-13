# c14 -- independent verification audit

The baseline run reports `backends.independent_cross_check: false`: `z3`, `dReal`
and `cvxpy` were all absent, so the native hybrid-zonotope BaB was the sole
implementation and its soundness rested entirely on adversarial sampling against
itself. That is the largest structural vulnerability on the record, because a
bug in the zonotope mechanics would be invisible to a falsifier that shares the
same mechanics.

This directory closes as much of that gap as is actually closable, and states
plainly which part is not.

`z3-solver 5.0.0` was installed for this audit. `dReal` and `cvxpy` remain
absent (dReal needs a build toolchain on Windows; cvxpy would only re-check the
dissipativity LP that scipy/HiGHS already called infeasible).

## What is being audited

The certificate that actually closes is the **monolithic margin** obligation
(`ARCHITECTURE.md` 7b): for a fixed prompt anchor, BaB decomposes the steering
box `[-rho, rho]^4` into sub-boxes and discharges each by proving a sound upper
bound `s_hi < 0` on the unsafe-logit margin. The union of discharged boxes is
the whole box, so the certificate is exactly the conjunction of those per-box
claims. Auditing it means auditing that conjunction, not a paraphrase of it.

The per-layer growth obligation (7a) is not audited: it certified no `gamma` at
any radius in the baseline, so there is no positive claim to check.

## Files

| file | role |
|---|---|
| `ibp_ref.py` | reference interval engine, transcribed from `toy_transformer.forward`, sharing no code with `src/bounds.py` |
| `mvf_ref.py` | second reference engine: mean-value form with interval forward-mode AD, keeps the 4-D threat structure |
| `selftest_ibp.py` | pins `ibp_ref` to torch at `rho=0` |
| `selftest_mvf.py` | pins `mvf_ref`'s value AND its interval Jacobian to torch autograd |
| `z3_primitives.py` | exact SMT audit of the prover's primitives |
| `sweep_layernorm.py` | fine-grained map of the variance-bracket cliff |
| `c14_crosscheck.py` | replay + falsification + coverage + both reference engines |

Both reference engines are validated before use, which matters: an unvalidated
"independent" witness that happens to agree proves nothing.

```
ibp_ref  : max |stream - torch| 3.6e-15, |logits - torch| 7.1e-15, |margin| 5.3e-15
mvf_ref  : max |margin - torch| 3.6e-15, |J_interval - J_autograd| 5.6e-16
```

The Jacobian check is the strong one -- it would catch a wrong derivative rule in
LayerNorm, softmax or attention, and it passes at machine precision.

## Result 1: the independent sound bounds are vacuous, for a measurable reason

Neither reference engine closes a single box that the prover closes.

| radius | MVF bound | IBP bound |
|---|---|---|
| 5.0e-03 | 3.1e+31 | 82.11 |
| 6.2e-04 | 6.0e+24 | 82.11 |
| 4.0e-05 | 1.2e+16 | 82.11 |
| 1.0e-05 | 2.0e+08 | 82.11 |

This is not a coding failure, it is the model. Both engines decorrelate, and
anything that decorrelates dies through layer 1. `ibp_ref` pins the number
directly: an outward padding of `1e-12` injected at the input emerges at the
logits as `4.7e-4`, an **amplification of 4.7e8** through two blocks. The IBP
column saturates at a constant 82.11 because the final LayerNorm's structural
bound `|u/s| <= sqrt(D)` clamps a stream that has already blown up -- the bound is
sound and completely uninformative.

The prover survives because it carries the 4 noise symbols exactly. So the
honest statement is: **an end-to-end independent sound bound is not achievable by
any decorrelating method on this model**, and the certificate at the whole-network
level still rests on the native engine. Confirming it end-to-end would require a
second correlation-preserving implementation (a CROWN/DeepPoly-style backward
linear relaxation), which is a reimplementation project, not an audit.

## Result 2: the exact solver does confirm the primitives

z3 has no wrapping problem -- it is exact -- so it has traction where interval
engines do not, on the primitives the end-to-end proof is assembled from.

At `rho=0.02`, on 8 boxes drawn from the 128 the prover actually discharges:

| check | expected | result |
|---|---|---|
| margin readout, soundness | `unsat` | **8/8 unsat** |
| margin readout, tightness | `sat` | **7/8 sat**, 1 timeout |
| relu containment | `unsat` | 0/3 conclusive (all timeout) |
| layernorm containment | `unsat` | 0/2 conclusive (all timeout) |

The readout is the component that carried the patched bug, and it is the one
that came back conclusive: every bound checked is confirmed sound, and 7 of 8
are confirmed tight. ReLU containment (128 units x 100 symbols, with an `If` per
unit) and LayerNorm containment (exact `1/sqrt(var+eps)` over 32 coordinates)
both exceed the 180s budget; those remain unconfirmed rather than refuted.

Separately, `c14_crosscheck.py` found **no unsoundness** at any radius: across
3209 discharged boxes, 2048 samples each plus 30-step PGD confined to each box,
no sampled point ever exceeded the bound claimed for its own box, and the true
margin never rose above `-11.337`.

The margin query uses the identity

```
max_u L_u - max_j L_j > B   <=>   EXISTS u . FORALL j . L_u - L_j > B
```

which keeps it in linear real arithmetic. Encoding the two maxima with nested
`If` instead makes z3 branch over 20 cases and return `unknown`; this form is
decided. Two queries are run per box:

- **soundness** (`unsat` expected): no point of the logit form exceeds the bound.
- **tightness** (`sat` expected): some point comes within `1e-3` of it.

The tightness query is the one that matters for the patched readout bug. The
original `max_u L_u - min_j L_j` is *still a valid upper bound*, so a soundness
check passes on the buggy code; it is only inflated by ~11.7 logits, and only a
tightness check sees that. A regression suite that checks soundness alone would
not have caught it.

These queries are near-degenerate LPs -- the prover's bound is the exact
supremum, so refuting it exercises exact rational simplex over ~128 box
variables. Measured 13-18s each and erratic; a 30s timeout returns `unknown`
everywhere and looks like solver failure. It is not. Budget 180s.

`unknown` in the output is a timeout, never a refutation.

## Result 3: the LayerNorm attribution is backwards

`results/c14_ln_sweep.json`, 36 radii. Tracking the layer-1 `ln2` site:

| rho | margin bound | unstable ReLUs | LN spread (max) |
|---|---|---|---|
| 0.00286 | -1.06e+01 | 6/640 | 1.025 |
| 0.00447 | -8.92e+00 | 23/640 | 1.064 |
| 0.00559 | -5.91e+00 | 44/640 | 1.112 |
| **0.00699** | **+9.63e+00** | 77/640 | **1.218** |
| 0.00874 | +1.00e+03 | 116/640 | 1.493 |
| 0.01710 | +1.32e+08 | 128/640 | 3.12e+03 |
| 0.02674 | +2.15e+10 | 128/640 | 4.71e+05 |

The unsplit bound crosses zero at `rho ~ 0.0070`. At that radius the LayerNorm
variance bracket is still **essentially tight** (spread 1.22, and the `eps` floor
is not engaged at all). The bracket only degrades at `rho >= 0.0171`, by which
point the bound has already passed `+1e8`.

So the causal order is:

> unstable ReLU count explodes  ->  bound fails  ->  *later* the LayerNorm
> bracket goes off a cliff

not "LayerNorm bracket cliffs cause the conservativeness gap". The ReLU
relaxation is the primary driver; the LN cliff is a downstream consequence that
arrives after the certificate has already been lost. `verifier.py:254` has the
upstream half right (splitting reduces the unstable-ReLU count) but its "and
hence the bound" clause overstates LayerNorm's role at the radii that matter.

Two caveats worth keeping:

- These are **unsplit** bounds. The certificate reaches `rho=0.04` precisely
  because BaB splits into 2047 boxes, each with a small effective radius where
  unstable ReLU counts are low. The sweep maps the mechanism, not the certificate.
- The bound is **not monotone** in `rho` (e.g. `2.1e10` at 0.0267 but `3.5e4` at
  0.0334). `promote_E_topk` and `compact` are radius-dependent heuristics, so a
  larger box can get a luckier promotion. Any claim of the form "the bound
  degrades smoothly with radius" is false as stated.

## Scope note on O(L)

The `O(L)` property in `ARCHITECTURE.md` is a statement about **cost** -- each
layer's obligation is discharged independently over a shared box -- not about
tightness. Since the compositional growth route certified no `gamma` at any
radius, there is no radius at which an O(L) *tightness* claim collapses; it was
never established. What `sweep_layernorm.py` localises is the bracket cliff
driving the monolithic bound, which is the certificate that actually closes.

## Running

```bash
python audit/selftest_ibp.py                 # validate reference engine 1
python audit/selftest_mvf.py                 # validate reference engine 2
python audit/sweep_layernorm.py              # ~4 s
python audit/z3_primitives.py --rho 0.02     # slow: ~200 s per box
python audit/c14_crosscheck.py               # falsification + coverage
```
