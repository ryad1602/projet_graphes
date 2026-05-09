"""
FunSearch implementation for GraphBench - Part 2.
Uses the Anthropic API to evolve better score functions for the heuristic search.
"""
import random
import time
import json
import requests
import networkx as nx
import numpy as np
from conjecture import load_benchmark, check_graph_class, to_graph6
from solver_v2 import fast_compute, make_initial_population, GENERAL_MUTS, CLAW_FREE_MUTS, TREE_MUTS, repair, random_connected, random_tree, random_claw_free


# ──────────────────────────────────────────────────────────────
# Base heuristic score (starting point)
# ──────────────────────────────────────────────────────────────

BASE_SCORE_CODE = '''
def heuristic_score(G, invariants, conjecture):
    """
    G : networkx graph
    invariants : dict of computed invariants
    conjecture : Conjecture object
    Returns a float score to maximize.
    """
    violation = conjecture.violation(invariants)
    n = invariants.get("order", 0)
    m = invariants.get("size", 0)
    delta = invariants.get("minimum_degree", 0)
    Delta = invariants.get("maximum_degree", 0)
    diam = invariants.get("diameter", 0)
    gamma = invariants.get("domination_number", 0)
    alpha = invariants.get("independence_number", 0)
    tau = invariants.get("vertex_cover_number", 0)
    triangles = invariants.get("triangle_number", 0)
    density = 0
    if n > 1:
        density = 2 * m / (n * (n - 1))
    return (
        10.0 * violation
        + 0.3 * diam
        + 0.2 * Delta
        + 0.1 * triangles
        - 0.05 * n
        - 0.2 * abs(density - 0.5)
    )
'''


# ──────────────────────────────────────────────────────────────
# Evaluate a score function on a sample of conjectures
# ──────────────────────────────────────────────────────────────

def evaluate_score_function(score_code, conjectures_sample, time_per_conj=8):
    """
    Evaluate a heuristic_score function by running the search on a sample.
    Returns: total_cost (lower is better), success_rate.
    """
    # Compile the score function
    try:
        namespace = {}
        exec(score_code, namespace)
        score_fn = namespace['heuristic_score']
    except Exception as e:
        return float('inf'), 0, str(e)

    total_cost = 0
    found = 0

    for conj in conjectures_sample:
        result = search_with_score_fn(conj, score_fn, time_limit=time_per_conj)
        graph, inv, violation, elapsed = result
        if violation > 1e-9:
            found += 1
            total_cost += elapsed
        else:
            total_cost += 120  # penalty

    success_rate = found / len(conjectures_sample) if conjectures_sample else 0
    return total_cost, success_rate, None


def search_with_score_fn(conjecture, score_fn, time_limit=8):
    """Run search using the given heuristic score function."""
    start = time.time()
    subgroup = conjecture.subgroup
    needed = conjecture.required_invariant_names()
    # Also compute some extra invariants for the score function
    all_needed = needed | {'order', 'size', 'minimum_degree', 'maximum_degree',
                           'diameter', 'domination_number', 'independence_number',
                           'vertex_cover_number', 'triangle_number'}

    if 'tree' in subgroup:
        muts = TREE_MUTS
    elif 'claw_free' in subgroup:
        muts = CLAW_FREE_MUTS
    else:
        muts = GENERAL_MUTS

    population = make_initial_population(conjecture, size=20)
    best_graph = None
    best_violation = float('-inf')
    best_inv = None
    pool = []

    for G in population:
        if time.time() - start > time_limit:
            break
        if not check_graph_class(G, subgroup):
            continue
        try:
            inv = fast_compute(G, all_needed)
            violation = conjecture.violation(inv)
            score = score_fn(G, inv, conjecture)
            pool.append((score, G, inv, violation))
            if violation > best_violation:
                best_violation = violation
                best_graph = G
                best_inv = inv
                if violation > 1e-9:
                    return G, inv, violation, time.time() - start
        except:
            pass

    pool.sort(key=lambda x: x[0], reverse=True)
    pool = pool[:20]

    while time.time() - start < time_limit:
        if pool:
            # Select from top candidates
            _, G, _, _ = pool[random.randint(0, min(4, len(pool)-1))]
        else:
            n = random.randint(5, 30)
            try:
                if 'tree' in subgroup:
                    G = random_tree(n)
                elif 'claw_free' in subgroup:
                    G = random_claw_free(n)
                else:
                    G = random_connected(n)
            except:
                continue

        H = G.copy()
        for _ in range(random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]):
            try:
                H = random.choice(muts)(H)
            except:
                pass

        try:
            H = repair(H, subgroup)
        except:
            continue

        if H.number_of_nodes() < 3 or H.number_of_nodes() > 50:
            continue
        if not check_graph_class(H, subgroup):
            continue

        try:
            inv = fast_compute(H, all_needed)
            violation = conjecture.violation(inv)
            score = score_fn(H, inv, conjecture)

            if violation > best_violation:
                best_violation = violation
                best_graph = H
                best_inv = inv
                if violation > 1e-9:
                    return H, inv, violation, time.time() - start

            pool.append((score, H, inv, violation))
        except:
            continue

        if len(pool) > 40:
            pool.sort(key=lambda x: x[0], reverse=True)
            pool = pool[:20]

    return best_graph, best_inv, best_violation, time.time() - start


# ──────────────────────────────────────────────────────────────
# LLM interface (Anthropic API)
# ──────────────────────────────────────────────────────────────

def call_llm(prompt, model="claude-sonnet-4-20250514"):
    """Call the Anthropic API to generate a new score function."""
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        data = response.json()
        text = "".join(block.get("text", "") for block in data.get("content", []))
        return text
    except Exception as e:
        return None


def extract_code(text):
    """Extract Python code from LLM response."""
    if text is None:
        return None
    # Try to extract code between ```python and ```
    import re
    patterns = [
        r'```python\s*(.*?)```',
        r'```\s*(def heuristic_score.*?)```',
        r'(def heuristic_score.*?)(?:\n\n|\Z)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if 'heuristic_score' in code:
                return code
    # If whole response is code
    if 'def heuristic_score' in text:
        start = text.index('def heuristic_score')
        return text[start:].strip()
    return None


def build_funsearch_prompt(best_functions, conjectures_sample, iteration):
    """Build a prompt for FunSearch iteration."""
    conjecture_examples = []
    for c in conjectures_sample[:5]:
        conjecture_examples.append(
            f"  - Conjecture {c.id}: {c.x_name} {c.sign} f({c.y_name}), classes: {c.subgroup}"
        )

    best_fn_text = ""
    for i, (score, fn_code, success_rate) in enumerate(best_functions[:3]):
        best_fn_text += f"\n# Function {i+1} — found {success_rate*100:.0f}% of counterexamples:\n{fn_code}\n"

    prompt = f"""You are helping improve a graph theory conjecture refutation system (GraphBench).

The system searches for counterexamples to graph theory conjectures using a heuristic score function.
A conjecture has form: y(G) <= f(x(G)) or y(G) >= f(x(G)) where x,y are graph invariants.

AVAILABLE INVARIANTS in the `invariants` dict:
- order (n), size (m), density, minimum_degree (δ), maximum_degree (Δ)
- average_degree, diameter, radius, clique_number, triangle_number
- domination_number (γ), total_domination_number, independence_number (α)
- vertex_cover_number (τ), independent_domination_number, matching_number (μ)
- randic_index, harmonic_index, first_zagreb_index, second_zagreb_index
- proximity, remoteness, largest_eigenvalue, largest_distance_eigenvalue
- second_smallest_laplace_eigenvalue

The `conjecture` object has:
- conjecture.violation(invariants) → float (positive means counterexample found!)
- conjecture.sign ('<=', '>=')
- conjecture.x_name, conjecture.y_name
- conjecture.subgroup (list: 'connected', 'tree', 'claw_free')

Example conjectures being searched:
{chr(10).join(conjecture_examples)}

ITERATION {iteration} — Here are the best functions found so far:
{best_fn_text}

Your task: write a BETTER `heuristic_score` function that guides the search more effectively.
The score function should:
1. Return a high value when violation > 0 (always weight violation heavily)
2. Add a bonus/penalty that guides toward graphs likely to violate the conjecture
3. Consider the graph structure (size, density, degree distribution, etc.)
4. Be class-aware (e.g., for trees, prioritize different structures than for dense graphs)

Key insight: to violate "y <= f(x)", we need y large and f(x) small.
To violate "y >= f(x)", we need y small and f(x) large.

Write ONLY the Python function, no explanation. Use exactly this signature:

def heuristic_score(G, invariants, conjecture):
    violation = conjecture.violation(invariants)
    # ... your code here ...
    return <score>
"""
    return prompt


# ──────────────────────────────────────────────────────────────
# FunSearch main loop
# ──────────────────────────────────────────────────────────────

def funsearch(conjectures, n_iterations=5, sample_size=10, time_per_eval=5, verbose=True):
    """
    FunSearch: iteratively evolve the heuristic score function using LLM.

    Returns: best_score_code, best_cost, history
    """
    # Sample a diverse set of conjectures for evaluation
    sample = random.sample(conjectures, min(sample_size, len(conjectures)))

    # Start with the base function
    best_functions = []
    base_cost, base_rate, err = evaluate_score_function(BASE_SCORE_CODE, sample, time_per_eval)

    best_functions.append((base_cost, BASE_SCORE_CODE, base_rate))
    best_cost = base_cost
    best_code = BASE_SCORE_CODE

    history = [{'iteration': 0, 'cost': base_cost, 'success_rate': base_rate, 'code': 'base'}]

    if verbose:
        print(f"[FunSearch] Iteration 0 (base): cost={base_cost:.1f}, success={base_rate*100:.0f}%")

    for iteration in range(1, n_iterations + 1):
        # Build prompt with best functions
        sorted_fns = sorted(best_functions, key=lambda x: x[0])
        prompt = build_funsearch_prompt(sorted_fns[:3], sample, iteration)

        # Call LLM
        if verbose:
            print(f"[FunSearch] Iteration {iteration}: calling LLM...")
        response = call_llm(prompt)
        new_code = extract_code(response)

        if new_code is None:
            if verbose:
                print(f"  -> LLM returned no valid code, skipping")
            continue

        # Evaluate new function
        if verbose:
            print(f"  -> Evaluating new score function...")
        try:
            cost, rate, err = evaluate_score_function(new_code, sample, time_per_eval)
        except Exception as e:
            if verbose:
                print(f"  -> Evaluation error: {e}")
            continue

        best_functions.append((cost, new_code, rate))
        history.append({
            'iteration': iteration,
            'cost': cost,
            'success_rate': rate,
            'code': new_code
        })

        if verbose:
            print(f"  -> cost={cost:.1f}, success={rate*100:.0f}% | prev_best={best_cost:.1f}")

        if cost < best_cost:
            best_cost = cost
            best_code = new_code
            if verbose:
                print(f"  -> NEW BEST! cost={cost:.1f}")

    # Final run with best function on ALL conjectures
    if verbose:
        print(f"\n[FunSearch] Final evaluation with best function on all {len(conjectures)} conjectures...")

    return best_code, best_cost, history


# ──────────────────────────────────────────────────────────────
# Run FunSearch + full evaluation
# ──────────────────────────────────────────────────────────────

def run_funsearch_benchmark(benchmark_path='benchmark/benchmark.xlsx',
                             n_iterations=5, time_per_eval=5, verbose=True):
    conjectures = load_benchmark(benchmark_path)

    print("=== FunSearch Phase ===")
    best_code, best_cost, history = funsearch(
        conjectures,
        n_iterations=n_iterations,
        sample_size=15,
        time_per_eval=time_per_eval,
        verbose=verbose
    )

    print(f"\nBest function code:\n{best_code}")

    # Save history
    with open('results/funsearch_history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)

    return best_code, history


if __name__ == '__main__':
    best_code, history = run_funsearch_benchmark(
        n_iterations=5,
        time_per_eval=6,
        verbose=True
    )
    print("\nFunSearch complete.")
    print(f"History: {len(history)} iterations")
    print(f"Best cost over sample: {min(h['cost'] for h in history):.1f}")
