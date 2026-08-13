"""Structurally constrained compositional Neural Lyapunov certificates for
transformer safety.

Import side effect: pins the OpenMP / thread environment BEFORE torch is
imported anywhere. The Anaconda build on this machine ships a duplicate
libiomp5md.dll and aborts the interpreter otherwise.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

__all__ = [
    "task",
    "toy_transformer",
    "frames",
    "sae",
    "icnn",
    "bounds",
    "verifier",
    "dissipativity",
    "soundness",
]
