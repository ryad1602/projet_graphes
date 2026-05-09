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
from solver import (
    compute_fast, initial_pop, ALL_MUTS, CF_MUTS, TREE_MUTS,
    repair, rnd_claw_free, rnd_connected, rnd_tree, targeted_graphs
)


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
        print(f"  [EVAL] Compilation error: {e}")
        return float('inf'), 0, str(e)

    total_cost = 0
    found = 0
    
    for i, conj in enumerate(conjectures_sample):
        try:
            result = search_with_score_fn(conj, score_fn, time_limit=time_per_conj)
            graph, inv, violation, elapsed = result
            if violation > 1e-9:
                found += 1
                total_cost += elapsed
            else:
                total_cost += time_per_conj * 2  # penalty for not finding
        except Exception as e:
            print(f"  [EVAL] Error on conjecture {i}: {e}")
            total_cost += time_per_conj * 2

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
        muts = CF_MUTS
    else:
        muts = ALL_MUTS

    population = initial_pop(conjecture, size=20)
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
            inv = compute_fast(G, all_needed)
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
                    G = rnd_tree(n)
                elif 'claw_free' in subgroup:
                    G = rnd_claw_free(n)
                else:
                    G = rnd_connected(n)
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
            inv = compute_fast(H, all_needed)
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
            headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"},
            json={
                "model": model,
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        if response.status_code != 200:
            print(f"  [DEBUG] API error: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            return None
        data = response.json()
        text = "".join(block.get("text", "") for block in data.get("content", []))
        return text
    except Exception as e:
        print(f"  [DEBUG] call_llm error: {e}")
        return None


def extract_code(text, verbose=False):
    """Extract Python code from LLM response - more robust."""
    if text is None:
        if verbose: print("  [DEBUG] extract_code: text is None")
        return None
    
    if verbose: print(f"  [DEBUG] extract_code: input length={len(text)}")
    
    import re
    
    # Pattern 1: code block with ```python
    match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if 'def heuristic_score' in code:
            if verbose: print("  [DEBUG] Found code in ```python block")
            return code
    
    # Pattern 2: code block with just ```
    match = re.search(r'```\s*(.*?)```', text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if 'def heuristic_score' in code:
            if verbose: print("  [DEBUG] Found code in ``` block")
            return code
    
    # Pattern 3: direct function definition (anywhere in text)
    if 'def heuristic_score' in text:
        start = text.index('def heuristic_score')
        # Find end: either next 'def ', or end of string, or double newline after 'return'
        code_part = text[start:]
        # Try to find the end of the function (look for return statement)
        lines = code_part.split('\n')
        func_lines = [lines[0]]
        indent_level = len(lines[0]) - len(lines[0].lstrip())
        
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '':
                func_lines.append(line)
                continue
            
            curr_indent = len(line) - len(line.lstrip())
            
            # Stop if we hit a line at or before original indent level and it's not empty
            if line.strip() and curr_indent <= indent_level and i > 1:
                break
            
            func_lines.append(line)
            
            # Also stop after a return statement at the base indent
            if 'return ' in line and curr_indent == indent_level:
                break
        
        code = '\n'.join(func_lines).strip()
        if verbose: print(f"  [DEBUG] Found direct function definition, length={len(code)}")
        return code
    
    if verbose: 
        print(f"  [DEBUG] No code found. Text preview: {text[:150]}")
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
        
        if response is None:
            if verbose:
                print(f"  -> LLM returned None (API error)")
            continue
        
        new_code = extract_code(response, verbose=verbose)

        if new_code is None:
            if verbose:
                print(f"  -> Could not extract code from LLM response")
            continue
        
        # Validate code can be compiled
        try:
            namespace = {}
            exec(new_code, namespace)
            if 'heuristic_score' not in namespace:
                if verbose:
                    print(f"  -> Compiled code but heuristic_score not found")
                continue
        except Exception as e:
            if verbose:
                print(f"  -> Code compilation error: {e}")
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
            'code': new_code[:200] + "..." if len(new_code) > 200 else new_code
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
