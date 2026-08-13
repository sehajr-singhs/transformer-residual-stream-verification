"""c15 -- verify the reproducibility bundle is complete and in sync.

Three checks:

  PRESENCE   every file the bundle claims to contain exists.
  SYNC       every GENERATED document is byte-identical to what its generator
             produces right now. This is the check that catches a hand-edit to
             ARCHITECTURE.md or C14_AUDIT.md, which would otherwise silently
             diverge from results/*.json.
  INTEGRITY  every checkpoint and result file still hashes to what
             results/environment.json recorded.

Exit code is nonzero if anything fails, so this can gate a release.
"""
import os, sys, json, hashlib, subprocess, tempfile, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

REQUIRED = [
    "README.md", "ARCHITECTURE.md", "C14_AUDIT.md", "requirements.txt",
    "src/bounds.py", "src/verifier.py", "src/report.py", "src/icnn.py",
    "src/sae.py", "src/task.py", "src/toy_transformer.py", "src/frames.py",
    "src/dissipativity.py", "src/soundness.py",
    "stages/a0_bootstrap.py",
    "tests/test_alignment.py", "tests/test_margin_exact.py", "tests/smoke.py",
    "audit/README.md", "audit/ibp_ref.py", "audit/mvf_ref.py",
    "audit/selftest_ibp.py", "audit/selftest_mvf.py", "audit/z3_primitives.py",
    "audit/sweep_layernorm.py", "audit/c14_crosscheck.py",
    "audit/report_c14.py", "audit/make_manuscript_table.py",
    "audit/lock_environment.py", "audit/recover_z3_json.py",
    "audit/verify_bundle.py",
    "audit/crown_reference.py", "audit/selftest_crown.py",
    "audit/crown_stagewise.py", "audit/c16_experiments.py",
    "audit/report_manuscript.py",
    "MANUSCRIPT.md",
    "results/baseline.json", "results/c14_audit.json", "results/c14_z3.json",
    "results/c14_ln_sweep.json", "results/manuscript_table.tex",
    "results/environment.json", "results/c16.json", "results/c16_crown.json",
    "results/c16_branching.json", "audit/recover_branching.py",
    "audit/c17_experiments.py", "results/c17.json",
    "audit/c18_variants.py", "results/c18.json",
    "audit/c19_trainability.py", "results/c19.json",
    "audit/c20_expressivity.py", "results/c20.json",
    "audit/c21_pareto_scaling.py", "results/c21.json",
    "audit/c22_asymptotic.py",
    # c22_ceiling.json is load-bearing: MANUSCRIPT.md section 6h derives the
    # abandonment argument from it. results/c22.json is deliberately absent --
    # that sweep was killed and the manuscript claims no such file.
    "results/c22_ceiling.json",
    "audit/c23_language.py", "results/c23.json", "data/tinyshakespeare.txt",
    # c24/c25/c26. These sweeps write one file PER CELL rather than one summary
    # JSON, so the parts directories are listed instead of a single result and
    # audit/c2425_data.py is the only reader of them.
    "audit/c24_scaling.py", "results/c24_parts",
    "audit/c25_certified.py", "audit/c25_arch.py",
    "results/c25_certified_parts", "results/c25_control_parts",
    "audit/c2425_data.py",
    "src/torch_bounds.py", "audit/selftest_torch_bounds.py",
    "checkpoints/model.pt", "checkpoints/sae.pt", "checkpoints/lyap.pt",
]

# generated document -> generator script
GENERATED = {
    "ARCHITECTURE.md": "src/report.py",
    "C14_AUDIT.md": "audit/report_c14.py",
    "results/manuscript_table.tex": "audit/make_manuscript_table.py",
    "MANUSCRIPT.md": "audit/report_manuscript.py",
}

fails = []


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


print("PRESENCE")
missing = [r for r in REQUIRED if not os.path.exists(os.path.join(ROOT, r))]
for m in missing:
    print(f"  MISSING  {m}")
    fails.append(f"missing {m}")
print(f"  {len(REQUIRED) - len(missing)}/{len(REQUIRED)} present")

print("\nSYNC (regenerate and compare)")
for doc, gen in GENERATED.items():
    p = os.path.join(ROOT, doc)
    if not os.path.exists(p):
        continue
    before = sha(p)
    r = subprocess.run([PY, os.path.join(ROOT, gen)], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  GENERATOR FAILED  {gen}: {r.stderr.strip()[:120]}")
        fails.append(f"generator failed {gen}")
        continue
    after = sha(p)
    ok = before == after
    print(f"  {'OK      ' if ok else 'DRIFTED '} {doc}  (from {gen})")
    if not ok:
        fails.append(f"{doc} was not in sync with {gen}")

print("\nINTEGRITY (against results/environment.json)")
envp = os.path.join(ROOT, "results", "environment.json")
if os.path.exists(envp):
    env = json.load(open(envp))
    n_ok = 0
    for rel, rec in env.get("artifact_hashes", {}).items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print(f"  MISSING  {rel}")
            fails.append(f"hashed artifact missing: {rel}")
            continue
        if rel in GENERATED or rel.endswith(".log"):
            continue          # regenerated above; hash legitimately moves
        if sha(p) != rec["sha256"]:
            print(f"  CHANGED  {rel}")
            fails.append(f"artifact changed since lock: {rel}")
        else:
            n_ok += 1
    print(f"  {n_ok} artifacts match their recorded hash")
    b = env["backends"]
    print(f"  backends: hzono={b['native_hzono_bab']} z3={b['z3']} "
          f"dreal={b['dreal']} cvxpy={b['cvxpy']}")
else:
    fails.append("results/environment.json absent")

print("\n" + "=" * 60)
if fails:
    print(f"BUNDLE NOT CLEAN -- {len(fails)} problem(s):")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("BUNDLE CLEAN: present, in sync with generators, hashes intact.")
