"""
PHASE 3: Appliquer la meilleure heuristique trouvée par FunSearch
sur TOUTES les 100 conjectures pour voir l'amélioration réelle
"""
import json
import os
import sys
import time
from conjecture import load_benchmark, to_graph6
from funsearch import search_with_score_fn
import concurrent.futures

RESULTS_DIR = 'results'
FAIL_PENALTY = 120


def run_with_heuristic(conj, heuristic_code, time_limit=60):
    """
    Lance search_with_score_fn avec une heuristique personnalisée.
    Retourne (graph, invariants, violation, elapsed_time, status)
    """
    # Compiler la heuristique
    try:
        namespace = {}
        exec(heuristic_code, namespace)
        score_fn = namespace['heuristic_score']
    except Exception as e:
        print(f"  ✗ Erreur compilation heuristique: {e}")
        return None, {}, float('-inf'), 0, "ERR"
    
    # Lancer la recherche
    hard_limit = int(time_limit * 1.3) + 5
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = ex.submit(search_with_score_fn, conj, score_fn, time_limit)
    
    try:
        graph, inv, violation, elapsed = future.result(timeout=hard_limit)
        status = "OK" if violation > 1e-9 else "FAIL"
        return graph, inv, violation, elapsed, status
    except concurrent.futures.TimeoutError:
        return None, {}, float('-inf'), float(hard_limit), "TO"
    except Exception as e:
        print(f"  ✗ Erreur: {e}")
        return None, {}, float('-inf'), 0, "ERR"
    finally:
        ex.shutdown(wait=False)


def apply_best_heuristic(heuristic_code=None, time_limit=60, verbose=True):
    """
    Applique la meilleure heuristique de FunSearch sur TOUS les conjectures.
    """
    # Charger l'historique FunSearch
    history_path = os.path.join(RESULTS_DIR, 'funsearch_history.json')
    
    if heuristic_code is None:
        if not os.path.exists(history_path):
            print("✗ Pas de funsearch_history.json trouvé!")
            print("  Lancer d'abord: python src/funsearch.py")
            return None
        
        with open(history_path) as f:
            history = json.load(f)
        
        # Prendre la meilleure fonction (celle avec le plus petit cost)
        best_iter = min(history, key=lambda x: x['cost'])
        heuristic_code = best_iter['code']
        
        print(f"✓ Chargée meilleure heuristique de l'itération {best_iter['iteration']}")
        print(f"  Cost: {best_iter['cost']:.1f}, Success rate: {best_iter.get('success_rate', 'N/A')}")
    
    # Charger tous les conjectures
    conjectures = load_benchmark('benchmark/benchmark.xlsx')
    
    print(f"\n{'='*70}")
    print(f"PHASE 3: Appliquer meilleure heuristique sur {len(conjectures)} conjectures")
    print(f"{'='*70}\n")
    
    results = []
    found = 0
    total_score = 0.0
    wall_start = time.time()
    
    for i, conj in enumerate(conjectures):
        graph, inv, violation, elapsed, status = run_with_heuristic(
            conj, heuristic_code, time_limit
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
        
        if verbose and (i + 1) % 5 == 0:
            wall = time.time() - wall_start
            print(
                f"[{i+1:3d}/{len(conjectures)}] Found: {found:3d} | "
                f"Score: {total_score:8.1f} | Wall: {wall:6.0f}s"
            )
    
    # Sauvegarder résultats
    out_path = os.path.join(RESULTS_DIR, 'results_optimized.json')
    output = {
        'total_score': round(total_score, 2),
        'found': found,
        'total_done': len(conjectures),
        'total': len(conjectures),
        'results': results,
        'heuristic_used': 'best_from_funsearch',
    }
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    wall_total = time.time() - wall_start
    
    print(f"\n{'='*70}")
    print(f"RÉSULTATS PHASE 3 (avec meilleure heuristique)")
    print(f"{'='*70}")
    print(f"Trouvées: {found}/{len(conjectures)}")
    print(f"Score total: {total_score:.1f}")
    print(f"Temps: {wall_total:.0f}s")
    print(f"Résultats sauvegardés: {out_path}")
    
    return results, total_score


def compare_results(verbose=True):
    """
    Compare les résultats baseline vs optimisés.
    """
    baseline_path = os.path.join(RESULTS_DIR, 'results.json')
    optimized_path = os.path.join(RESULTS_DIR, 'results_optimized.json')
    
    if not os.path.exists(baseline_path):
        print("✗ results.json non trouvé!")
        return
    
    if not os.path.exists(optimized_path):
        print("✗ results_optimized.json non trouvé!")
        return
    
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    with open(optimized_path) as f:
        optimized = json.load(f)
    
    baseline_score = baseline['total_score']
    baseline_found = baseline['found']
    
    optimized_score = optimized['total_score']
    optimized_found = optimized['found']
    
    improvement = baseline_score - optimized_score
    improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"COMPARAISON: BASELINE vs OPTIMISÉ")
    print(f"{'='*70}")
    print(f"\nScore total:")
    print(f"  Baseline:  {baseline_score:.1f}")
    print(f"  Optimisé:  {optimized_score:.1f}")
    print(f"  Gain:      {improvement:.1f} ({improvement_pct:+.1f}%)")
    print(f"\nConjectures résolues:")
    print(f"  Baseline:  {baseline_found}/{baseline['total_done']}")
    print(f"  Optimisé:  {optimized_found}/{optimized['total_done']}")
    print(f"  Gain:      {optimized_found - baseline_found:+d}")
    
    if improvement < 0:
        print(f"\n⚠️  La version optimisée est PIRE (plus lente)")
    elif improvement == 0:
        print(f"\n═ Aucune différence")
    else:
        print(f"\n✓ La version optimisée est MEILLEURE de {improvement:.1f} secondes!")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    print("Appliquer meilleure heuristique FunSearch sur tout le benchmark...")
    results, score = apply_best_heuristic(time_limit=60, verbose=True)
    
    print("\nComparaison avec baseline...")
    compare_results(verbose=True)
