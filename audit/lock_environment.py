"""c15 -- record the environment footprint that produced the artifacts.

Writes results/environment.json: interpreter, package versions, solver backends,
the seeds every stage uses, and a SHA-256 of each checkpoint and result file.

The hashes are the part that actually prevents artifact drift. A version pin
tells you what was installed; a checkpoint hash tells you whether the model the
numbers were computed from is still the model on disk.
"""
import os, sys, json, hashlib, platform, importlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "environment.json")

PKGS = ["numpy", "torch", "scipy", "z3", "dreal", "cvxpy", "matplotlib"]
# Seeds are hard-coded across stages rather than threaded through one config;
# recording them here is the only place they appear together.
SEEDS = {
    "stages/a0_bootstrap.py:SEED": 0,
    "sae feature pick (rng for U)": 11,
    "sae activation collection": 7,
    "task.enumerate_prompts anchor rng": 0,
    "audit/c14_crosscheck.py box subsample": 7,
    "audit/c14_crosscheck.py sampling": 99,
    "audit/c14_crosscheck.py coverage probe": 12345,
    "audit/z3_primitives.py box choice": 5,
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


versions = {}
for p in PKGS:
    try:
        m = importlib.import_module(p)
        v = getattr(m, "__version__", None)
        if p == "z3":
            v = m.get_version_string()
        versions[p] = v or "present (no __version__)"
    except Exception:
        versions[p] = None

files = {}
# `data` is hashed because c23 takes a corpus as input: a result computed from
# a different TinyShakespeare copy is a different result, and the version pin
# would not catch it.
for sub in ("checkpoints", "results", "data"):
    d = os.path.join(ROOT, sub)
    if not os.path.isdir(d):
        continue
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if os.path.isfile(p) and fn != "environment.json":
            files[f"{sub}/{fn}"] = {"sha256": sha256(p),
                                    "bytes": os.path.getsize(p)}

src_files = {}
for sub in ("src", "stages", "audit", "tests"):
    d = os.path.join(ROOT, sub)
    if not os.path.isdir(d):
        continue
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if os.path.isfile(p) and fn.endswith((".py", ".md")):
            src_files[f"{sub}/{fn}"] = sha256(p)[:16]

env = {
    "python": sys.version.split()[0],
    "executable": sys.executable,
    "platform": platform.platform(),
    "packages": versions,
    "backends": {
        "native_hzono_bab": True,
        "z3": versions.get("z3") is not None,
        "dreal": versions.get("dreal") is not None,
        "cvxpy": versions.get("cvxpy") is not None,
    },
    "seeds": SEEDS,
    "artifact_hashes": files,
    "source_hashes_truncated": src_files,
    "notes": [
        "KMP_DUPLICATE_LIB_OK is pinned in src/__init__.py before torch is "
        "imported; this Anaconda build ships a duplicate libiomp5md.dll.",
        "The model checkpoint was trained in float32. The verifier and both "
        "audit reference engines run in float64, so comparisons against torch "
        "must cast the model to double or they show ~1e-6 float32 residuals "
        "that are not engine error.",
        "z3-solver was installed during audit c14; the original baseline run "
        "predates it and correctly reports independent_cross_check: false.",
    ],
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(env, open(OUT, "w"), indent=2)
print(json.dumps({"python": env["python"], "packages": env["packages"],
                  "backends": env["backends"],
                  "artifacts_hashed": len(files),
                  "sources_hashed": len(src_files)}, indent=2))
print(f"wrote {OUT}")
