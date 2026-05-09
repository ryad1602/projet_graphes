"""
Verifier: check whether a graph6-encoded graph violates a benchmark conjecture.

Usage:
    python verifier.py <conjecture_id> <graph6_string>

Example:
    python verifier.py 1708 OdOGEC??G@_N?N??_?GCG
    python verifier.py 2051 "N?WUaA?_G?_?o?S??"
"""

import os
import sys
import networkx as nx

# ── path setup ────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from conjecture import load_benchmark
from solver import compute

BENCHMARK_PATH = os.path.join(_DIR, '..', 'benchmark', 'benchmark.xlsx')


# ── helpers ───────────────────────────────────────────────────

def decode_g6(g6_str: str) -> nx.Graph:
    """Decode a graph6 string into a NetworkX graph."""
    s = g6_str.strip()
    return nx.from_graph6_bytes(s.encode('ascii'))


def compute_invariants_exact(G: nx.Graph, needed: set) -> dict:
    """Compute the exact invariants needed for the conjecture."""
    return compute(G, needed)


def check_subgroup(G: nx.Graph, subgroup: list) -> tuple[bool, list]:
    """Return (ok, list_of_failed_classes)."""
    from conjecture import CLASS_CHECKERS
    failed = [cls for cls in subgroup if cls in CLASS_CHECKERS and not CLASS_CHECKERS[cls](G)]
    return len(failed) == 0, failed


# ── main audit ────────────────────────────────────────────────

def run_audit(c_id, g6_str: str):
    # 1. Load conjecture
    try:
        all_conjs = load_benchmark(BENCHMARK_PATH)
    except Exception as e:
        print(f"Cannot load benchmark: {e}")
        sys.exit(1)

    conj = next((c for c in all_conjs if str(c.id) == str(c_id)), None)
    if conj is None:
        print(f"Conjecture ID {c_id} not found in benchmark.")
        sys.exit(1)

    # 2. Decode graph
    try:
        G = decode_g6(g6_str)
    except Exception as e:
        print(f"Invalid graph6 string: {e}")
        sys.exit(1)

    n = G.number_of_nodes()
    m = G.number_of_edges()

    # 3. Check subgroup membership
    subgroup_ok, failed_classes = check_subgroup(G, conj.subgroup)

    # 4. Compute invariants (exact, via ILP where needed)
    needed = conj.required_invariant_names()
    try:
        invs = compute_invariants_exact(G, needed)
    except Exception as e:
        print(f"Error computing invariants: {e}")
        sys.exit(1)

    x_val = invs.get(conj.x_name)
    y_val = invs.get(conj.y_name)

    if x_val is None or y_val is None:
        print(f"Could not compute {conj.x_name} or {conj.y_name}.")
        sys.exit(1)

    bound = conj.evaluate_bound(x_val)
    viol  = conj.violation(invs)   # positive = counterexample

    # 5. Print report
    SEP  = "=" * 62
    SEP2 = "-" * 62
    print()
    print(SEP)
    print(f"  CONJECTURE {conj.id} AUDIT")
    print(SEP)
    print(f"  Formula  : {conj.y_name} {conj.sign} f({conj.x_name})")
    print(f"  Subgroup : {conj.subgroup}")
    print(f"  Bound    : {conj.text}")
    print(SEP2)
    print(f"  Graph    : {n} vertices, {m} edges")
    print(f"  g6       : {g6_str.strip()}")
    print(SEP2)

    # Subgroup check
    if not subgroup_ok:
        print(f"  SUBGROUP : FAIL — graph does not satisfy {failed_classes}")
    else:
        print(f"  Subgroup : OK ({conj.subgroup})")

    # Invariant values
    print(f"  X = {conj.x_name:<32} = {x_val}")
    print(f"  Y = {conj.y_name:<32} = {y_val}")
    print(f"  f(X) (bound)                         = {bound:.6f}")
    print(f"  Violation (positive = counterexample) = {viol:.6f}")
    print(SEP2)

    if not subgroup_ok:
        print("  VERDICT : INVALID — graph not in required subgroup")
    elif viol > 1e-9:
        print(f"  VERDICT : COUNTEREXAMPLE CONFIRMED  (violation = {viol:.6f})")
    elif abs(viol) <= 1e-9:
        print(f"  VERDICT : TIGHT (conjecture holds with equality)")
    else:
        print(f"  VERDICT : Conjecture satisfied  (gap = {-viol:.6f})")

    print(SEP)
    print()


# ── entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(0)

    c_id   = sys.argv[1]
    g6_str = sys.argv[2]
    run_audit(c_id, g6_str)
