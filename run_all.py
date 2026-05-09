"""
Lancer le workflow COMPLET: Baseline + FunSearch + Apply Best
"""
import os
import sys

# Ajouter clé API (remplace par ta vraie clé)
# os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."

print("=" * 70)
print("PHASE 1: Baseline (heuristique standard)")
print("=" * 70)
from src.main import run_benchmark
results, score = run_benchmark(time_limit=60, start_idx=0)
baseline_score = score

print("\n" + "=" * 70)
print("PHASE 2: FunSearch (générer et tester meilleures heuristiques)")
print("=" * 70)
from src.funsearch import run_funsearch_benchmark
best_code, history = run_funsearch_benchmark(
    n_iterations=5,
    time_per_eval=6,
    verbose=True
)

print("\n" + "=" * 70)
print("PHASE 3: Appliquer la meilleure heuristique sur TOUS les conjectures")
print("=" * 70)
from src.apply_best_heuristic import apply_best_heuristic, compare_results
optimized_results, optimized_score = apply_best_heuristic(time_limit=60, verbose=True)

print("\n" + "=" * 70)
print("RÉSUMÉ FINAL")
print("=" * 70)
print(f"\nBaseline score: {baseline_score:.1f}")
print(f"Score optimisé: {optimized_score:.1f}")
print(f"Amélioration: {baseline_score - optimized_score:.1f} secondes")

if optimized_score < baseline_score:
    improvement_pct = ((baseline_score - optimized_score) / baseline_score) * 100
    print(f"✓ FunSearch a amélioré de {improvement_pct:.1f}%!")
else:
    print(f"⚠️  Pas d'amélioration")

compare_results(verbose=True)
