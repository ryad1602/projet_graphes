"""
PHASE 3: Appliquer les meilleurs poids de mutations trouves par FunSearch
sur TOUTES les 100 conjectures pour voir l'amelioration reelle.

Charge results/best_mutations.py et injecte les poids dans solver.search().
Ecrase results.json seulement si le score est meilleur que le baseline.
"""
import json
import os
import sys
import time
import shutil
from conjecture import load_benchmark, to_graph6
from solver import search
import concurrent.futures

RESULTS_DIR = 'results'
FAIL_PENALTY = 120
_EMPTY_PATH = {"phase_found": None, "n_resets": 0, "sa_steps": 0, "milestones": []}


def _load_mutation_weights():
    """Charge best_mutations.py et retourne le dict MUTATION_WEIGHTS."""
    path = os.path.join(RESULTS_DIR, 'best_mutations.py')
    if not os.path.exists(path):
        print("Pas de results/best_mutations.py -- lancer d'abord : py src/funsearch.py")
        return None
    ns = {}
    with open(path) as f:
        exec(f.read(), ns)
    if 'MUTATION_WEIGHTS' not in ns:
        print("MUTATION_WEIGHTS non trouve dans best_mutations.py")
        return None
    return ns['MUTATION_WEIGHTS']


def run_with_mutation_weights(conj, mutation_weights, time_limit=60):
    """Lance solver.search() avec les poids de mutations FunSearch injectes."""
    hard_limit = int(time_limit * 1.3) + 5
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = ex.submit(search, conj, time_limit, mutation_weights=mutation_weights)
    try:
        graph, inv, violation, elapsed = future.result(timeout=hard_limit)
        status = "OK" if violation > 1e-9 else "FAIL"
        return graph, inv, violation, elapsed, status
    except concurrent.futures.TimeoutError:
        return None, {}, float('-inf'), float(hard_limit), "TO"
    except Exception as e:
        print(f"  Erreur: {e}")
        return None, {}, float('-inf'), 0, "ERR"
    finally:
        ex.shutdown(wait=False)


def apply_best_heuristic(time_limit=60, verbose=True):
    """
    Applique les meilleurs poids de mutations sur les 100 conjectures.
    """
    mutation_weights = _load_mutation_weights()
    if mutation_weights is None:
        return None

    print(f"Poids charges depuis results/best_mutations.py")
    print(f"  general   ({len(mutation_weights['general'])} mutations)")
    print(f"  tree      ({len(mutation_weights['tree'])} mutations)")
    print(f"  claw_free ({len(mutation_weights['claw_free'])} mutations)")

    conjectures = load_benchmark('benchmark/benchmark.xlsx')

    print(f"\n{'='*70}")
    print(f"PHASE 3 : poids de mutations FunSearch sur {len(conjectures)} conjectures")
    print(f"{'='*70}\n")

    results = []
    found = 0
    total_score = 0.0
    wall_start = time.time()

    for i, conj in enumerate(conjectures):
        graph, inv, violation, elapsed, status = run_with_mutation_weights(
            conj, mutation_weights, time_limit
        )

        score_pts = elapsed if status == "OK" else FAIL_PENALTY
        total_score += score_pts
        if status == "OK":
            found += 1

        result = {
            'conjecture_id': conj.id,
            'status': status,
            'g6': to_graph6(graph) if status == "OK" and graph else "",
            'violation': float(violation) if violation not in (float('-inf'), None) else -999,
            'time': round(elapsed, 3),
            'score_pts': round(score_pts, 3),
            'x_name': conj.x_name,
            'y_name': conj.y_name,
            'subgroup': conj.subgroup,
            'n_nodes': graph.number_of_nodes() if graph else 0,
        }
        results.append(result)

        if verbose:
            wall = time.time() - wall_start
            v_str = f"{violation:.4f}" if violation not in (float('-inf'), None) else "  -inf"
            print(
                f"[{i+1:3d}/{len(conjectures)}] {conj.id:5d} {status:4s} "
                f"v={v_str:>10}  t={elapsed:6.2f}s  pts={score_pts:6.1f}  "
                f"total={total_score:8.1f}  wall={wall:5.0f}s | {conj.x_name} vs {conj.y_name}"
            )

    out_path = os.path.join(RESULTS_DIR, 'results_optimized.json')
    output = {
        'total_score': round(total_score, 2),
        'found': found,
        'total_done': len(conjectures),
        'total': len(conjectures),
        'results': results,
        'source': 'funsearch_mutation_weights',
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    wall_total = time.time() - wall_start
    print(f"\n{'='*70}")
    print(f"RESULTATS PHASE 3 (poids de mutations FunSearch)")
    print(f"{'='*70}")
    print(f"Trouvees : {found}/{len(conjectures)}")
    print(f"Score    : {total_score:.1f}")
    print(f"Temps    : {wall_total:.0f}s")
    print(f"Sauvegarde -> {out_path}")

    return results, total_score


def compare_results():
    baseline_path = os.path.join(RESULTS_DIR, 'results.json')
    optimized_path = os.path.join(RESULTS_DIR, 'results_optimized.json')

    if not os.path.exists(baseline_path) or not os.path.exists(optimized_path):
        print("Fichiers manquants pour la comparaison.")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(optimized_path) as f:
        optimized = json.load(f)

    bs, bf = baseline['total_score'], baseline['found']
    os_, of = optimized['total_score'], optimized['found']
    gain = bs - os_
    pct = gain / bs * 100 if bs > 0 else 0

    print(f"\n{'='*70}")
    print(f"COMPARAISON : BASELINE vs POIDS DE MUTATIONS FUNSEARCH")
    print(f"{'='*70}")
    print(f"  Baseline  : {bs:.1f}s   ({bf}/100 trouvees)")
    print(f"  FunSearch : {os_:.1f}s   ({of}/100 trouvees)")
    print(f"  Gain      : {gain:+.1f}s  ({pct:+.1f}%)")

    if gain > 0:
        print(f"\n  FunSearch est MEILLEUR de {gain:.1f}s")
    elif gain < 0:
        print(f"\n  Baseline conservee ({bs:.1f} <= {os_:.1f})")
    else:
        print(f"\n  Aucune difference")
    print(f"{'='*70}\n")
    return gain


if __name__ == '__main__':
    results, score = apply_best_heuristic(time_limit=60, verbose=True)
    if results is None:
        sys.exit(1)

    gain = compare_results()

    if gain is not None and gain > 0:
        baseline_path = os.path.join(RESULTS_DIR, 'results.json')
        optimized_path = os.path.join(RESULTS_DIR, 'results_optimized.json')
        shutil.copy(optimized_path, baseline_path)
        print(f"results.json mis a jour avec FunSearch ({score:.1f}s)")
    else:
        print(f"Baseline conservee.")
