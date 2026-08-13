"""Pin down the generator-symbol alignment invariant.

`HZono.__add__` pairs generators by index. That is correct only when one
operand's symbol list is a prefix of the other's. `compact` reorders, so
compacting one side of a residual split breaks the pairing and yields a set that
is NOT an over-approximation -- a bound that looks sound and is not.

This test exhibits the failure directly, so nobody "optimizes" compaction back
into the middle of a block.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import bounds as bd

rng = np.random.default_rng(0)
n, B = 6, 1
lo = rng.uniform(-1, 0, size=(B, n)); hi = lo + rng.uniform(0.2, 1.0, size=(B, n))
W = rng.normal(size=(n, n))

z = bd.HZono.from_box(lo, hi)
branch = z.linear(W)                       # derived from z, same symbols

# CORRECT: prefix relationship preserved
good = z + branch
glo, ghi = good.bounds()

# WRONG: reorder one side only
zbad = z.compact(3)
bad = zbad + branch
blo, bhi = bad.bounds()

xs = lo + rng.random((40000, n)) * (hi - lo)
true = xs + xs @ W.T

viol_good = max((glo - true).max(), (true - ghi).max())
viol_bad = max((blo - true).max(), (true - bhi).max())

print(f"prefix-aligned add : max_violation = {viol_good:+.6e}  sound={viol_good <= 1e-9}")
print(f"reordered add      : max_violation = {viol_bad:+.6e}  sound={viol_bad <= 1e-9}")
assert viol_good <= 1e-9, "prefix-aligned addition must be sound"
assert viol_bad > 1e-9, ("reordered addition should be DEMONSTRABLY unsound; if this "
                         "no longer triggers, the invariant is being enforced elsewhere "
                         "and this test needs updating rather than deleting")

# The propagation path must never trip it: block() compacts only at boundaries.
print("alignment invariant OK")
