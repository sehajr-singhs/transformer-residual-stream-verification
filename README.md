# Certified Safety Margins for a Transformer Residual Stream

> **A strict Lyapunov decrease certificate is infeasible for this model — so the
> obligation is restated as bounded finite-horizon growth, proved rather than
> sampled, and the paper says exactly where that restatement stops helping.**

Structurally constrained, compositional neural Lyapunov certificates for
transformer safety. Proofs by construction over residual-stream dynamics, with
no post-hoc sampling or red-teaming in the claim path, and every number in
this repo read from a JSON result file, never hand-typed into a paper.

**Papers:** [Nature Machine Intelligence format](nmi_paper.pdf) ·
[IEEE format](ieee_paper.pdf) · [full manuscript](MANUSCRIPT.md) (source of
truth both papers are generated from) · [architecture report](ARCHITECTURE.md)
(generated from `results/baseline.json`, so it cannot drift from the run).

Both papers are drafted against the real `IEEEtran.cls` / `sn-jnl.cls`
templates and report their own gaps honestly, including one comparison this
project's local compute environment could not complete (Section III-C of the
IEEE paper) and one negative result from a schedule search run specifically to
try to close it (Section 6k of the manuscript). Neither gap is papered over.

## The bet

Treat the residual stream as a discrete-time nonlinear dynamical system with
the layer index as time,

```
x_{l+1} = x_l + Attn_l(LN(x_l)) + MLP_l(LN(x_l + Attn_l(LN(x_l))))
```

work in deviation coordinates `e_l = x_l - x*_l` around a nominal trajectory,
and formalize alignment safety as an invariant sublevel set of a
positive-definite `V` whose behaviour along layers is *proved* rather than
sampled.

## What survived contact with the mathematics

Two of the original design assumptions did not, and both are load-bearing:

1. **Strict decrease is infeasible, not just hard.** A positive-definite `V`
   with `V(e') <= gamma V(e)`, `gamma < 1`, exists **iff** the deviation
   dynamics has spectral radius below 1. Measured on the certified subspace it
   is **1.273**. No relaxation tightening or training schedule can produce
   that certificate. The obligation is therefore restated as bounded
   finite-horizon growth: certify the smallest provable `gamma_l` per layer,
   and compose. Still by construction, still sampling-free, and true.

2. **Small-gain composition cannot certify a residual stream.** The exact
   identity path forces `gamma_layer <= 1 + sum_h gamma_h + gamma_mlp >= 1`
   unconditionally. The measured cascade product is 5562.9. Contraction would
   have to come from sign-aware cancellation, which a norm gain discards by
   construction. The compositional payoff is not gain multiplication; it is
   that each layer's obligation is discharged independently over a shared box
   at O(L) cost.

## Measured results (nothing extrapolated)

**Certification.** By sound hybrid-zonotope branch-and-bound at 2048 boxes,
the certified radius reaches **rho = 0.04** on a 26,144-parameter, 2-layer
`d_model=32` transformer trained on a synthetic feature-routing task
(accuracy 1.000). Independent falsification across **3209 discharged boxes**
(2048 samples + 30-step PGD each) finds no violation and exact coverage of the
threat set (1.0000); z3 over exact rationals confirms the margin readout 8/8
sound, 7/8 tight, and is the check that actually caught a real readout bug (a
sound-but-loose `max-min` inflation of ~11.7 logits) that a soundness-only
regression test would have missed.

**Cross-engine check: partially closed.** Two from-scratch reference engines
(interval-bound, mean-value-form) are vacuous on this model — any
decorrelating abstraction amplifies **4.7e8x** through two blocks. A
CROWN-style engine that keeps the 4-dimensional steering structure exactly
still loses, and the loss is localized to softmax uncertainty entering as
interval coefficients: it tracks the prover to 1.61x at `L0 ln1` and diverges
to 39.5x (prover: 3.86x) at `L0 attn`. An independent auto_LiRPA/CROWN
baseline on the same trained model and threat model was attempted four times;
auto_LiRPA itself is verified working in this environment (a standalone
CROWN bound in <1s, and this project's own tail model wrapped in
`BoundedModule` returning a real bound in isolation), but the full comparison
never completed here across four separate attempts and a range of system
loads that rules out simple resource contention as the sole explanation. That
is reported as a documented attempt, not a fabricated number — see
`audit/c31_autolirpa_baseline.py` and Section III-C of `ieee_paper.pdf`.

**Scaling.** Replacing LayerNorm's data-dependent scale with a fixed affine
one delays the relaxation-gap wall rather than removing it: the fitted
width-scaling exponent falls from **11.12 to 2.46** (still super-linear) on
random initialisation, and the gain **survives training** at 5.3x on the one
grid cell certified-by-construction training converges on. On an unsaturated
task the exchange is a real Pareto frontier, not a free lunch: 17.1x
verifiability for 5.7 accuracy points at `d_model=8`; on real character-level
language modelling (TinyShakespeare, no accuracy ceiling to hide behind) the
verifiability gain grows **83.2x -> 879.3x** from `d_model=32` to 64 for a
1.78-1.82% perplexity cost. At depth `L=4` the standard variant is
uncertifiable at every width tested (gaps 1.25e10 to 5.82e22); the fixed-norm
variant stays certifiable throughout but the wall's onset does not follow a
smooth power law across the three widths measured.

**Certified training.** Putting a sound interval-bound term into the training
loss collapses the run-to-run spread of the certified gap from **1705x to
1.08x** at `L=2`, `d_model=32`, fixnorm, with zero unstable ReLUs on every
converged run — at a perplexity cost of 13.3-28.0%. This result is reported
for exactly the one cell where it holds: 18 of 18 other (depth, width, arm)
combinations diverged at every learning rate under the schedule used
throughout, and a further 9 of 9 runs diverged under three deliberately
gentler epsilon-ramp schedules run specifically to test whether that limit was
an artifact of one schedule choice. It was tested directly rather than left
open, and the answer at this compute budget is still divergence.

**Adversarial stress test.** A gradient-based search against the
differentiable training-signal engine finds real, repeatable OOD sensitivity —
a 16.9x relaxation-gap increase on repeated-token contexts, reproduced in 5 of
5 restarts. Checked black-box against the actual certifying engine (never
part of the gradient search), the same adversarial context shifts the sound
zonotope bound by only 1.05x, both points remaining far inside the certified
band with exact containment. One context pair; not a swept claim in either
direction.

No claim here has been demonstrated to transfer to a production model, and
none of the results above is offered as a general law of transformer
architectures. Every number is read from a `results/*.json` file by
`audit/report_manuscript.py`; nothing in `MANUSCRIPT.md` or either paper is
hand-typed.

## Layout

```
src/
  bounds.py         hybrid affine-form (zonotope + interval remainder) engine:
                    sound LayerNorm, linearized softmax, DeepZ ReLU, attention
  torch_bounds.py   differentiable interval-arithmetic engine (certified
                    training signal + the auto_LiRPA baseline's tail model)
  frames.py         invariant/relative coordinates; exact LayerNorm mean-gauge
  task.py           feature-routing task + unsafe-logit-margin safety spec
  toy_transformer.py 2-layer 4-head pre-LN model; numpy weight export
  sae.py            sparse autoencoder, bridge-gap accounting, support certificate
  icnn.py           ICNN Lyapunov function (PD by construction) + sound V bounds
  verifier.py       sound branch-and-bound in feature-coefficient space, CEGIS
  dissipativity.py  local supply rates, interconnection LP, identity-path result
  soundness.py      adversarial falsification of the prover's own bounds
  report.py         generates ARCHITECTURE.md from results/baseline.json
stages/
  a0_bootstrap.py   end-to-end pipeline -> results/baseline.json
audit/
  c14-c31_*.py      30 scripts: independent cross-checks (z3, from-scratch
                    IBP/MVF/CROWN engines, auto_LiRPA baseline), the character-
                    LM scaling study (6a-6l), certified training, adversarial
                    fuzzing, and the report/table generators
  report_manuscript.py   single source of truth: generates MANUSCRIPT.md
  verify_bundle.py       gates presence/sync/hash integrity of every result
tests/
  smoke.py          fast shape + soundness check; run this first
  test_alignment.py pins the generator-symbol alignment invariant
  diag_*.py         diagnostics that located each blow-up source
results/            every JSON the papers' numbers are read from
figs/               figures rendered from the measured data
```

## Running

The Anaconda build here ships a duplicate `libiomp5md.dll`; `src/__init__.py`
pins `KMP_DUPLICATE_LIB_OK` before torch is imported, so import `src` first.

```bash
pip install -r requirements.txt
python tests/smoke.py             # fast sanity + soundness
python tests/test_alignment.py    # the invariant that bit us
python stages/a0_bootstrap.py     # full pipeline (~15 min, CPU)
python src/report.py              # regenerate ARCHITECTURE.md
python audit/report_manuscript.py # regenerate MANUSCRIPT.md from results/
python audit/verify_bundle.py     # gate: every artifact present, synced, hashed
```

## Soundness discipline

- Everything under `certify_*` is sound: a `True` is a proof modulo floating
  point. `pgd_falsify*` is unsound by design and only ever produces
  counterexamples.
- `soundness.py` tries to break every bound by dense sampling of corners,
  faces, interior, and PGD witnesses. `max_violation <= 0` is the pass
  condition; a positive value is a bug, not a tolerance to widen.
- **The alignment invariant.** `HZono.__add__` pairs generators by index,
  which is valid only when one operand's symbol list is a prefix of the
  other's. `promote_E*` appends and is safe; `compact` reorders and may only
  be applied where a single lineage exists (block boundaries). Compacting
  mid-block silently pairs unrelated noise symbols and yields a bound that is
  not an over-approximation. `tests/test_alignment.py` exhibits the failure.
- **Independent cross-check: partially closed.** See "Measured results"
  above and `C14_AUDIT.md` / `audit/README.md` for the full account, including
  what remains open and why.

## License

MIT — see `LICENSE`.
