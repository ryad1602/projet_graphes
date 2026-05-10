---
name: GraphBench — contexte projet
description: Architecture, benchmark, état des conjectures, ce qui a été fait, workflow actuel
type: project
---

## Objectif

Trouver des contre-exemples à 100 conjectures (toutes REJECTED = fausses). Score = somme des temps d'exécution → minimiser. Fail/TO = pénalité 120 pts.

**How to apply:** Orienter sur vitesse de convergence + qualité population initiale.

---

## Structure du projet (état actuel)

```
projet_graphes/
├── benchmark/benchmark.xlsx      ← 100 conjectures (pandas)
├── src/
│   ├── main.py                   ← orchestrateur, ThreadPoolExecutor
│   ├── conjecture.py             ← parser + invariants (NE PAS MODIFIER)
│   ├── solver.py                 ← moteur heuristique (LE fichier clé)
│   ├── funsearch.py              ← Partie 2 : évolution heuristique sans API
│   ├── apply_best_heuristic.py   ← Phase 3 : relancer avec meilleure fonction
│   └── verifier.py               ← CLI: python verifier.py <id> <g6>
└── results/
    ├── results.json              ← résultats solver baseline
    ├── results_optimized.json    ← résultats avec heuristique FunSearch
    ├── funsearch_history.json    ← historique des itérations FunSearch
    └── best_heuristic.py         ← meilleure fonction de score trouvée
```

**Fichiers supprimés (debug inutiles) :** `search_clawfree.py`, `verify_seeds.py`

---

## État des conjectures

**Résultats typiques :** 94–100/100 trouvées par run, score 730–1200. Variance due au SA stochastique.

**bridge_star ajouté dans solver.py** (targeted_graphs + initial_pop) pour 1587/1891/1600/2120/2252. Ces conjectures ne devraient plus être des blocages systématiques.

**Conjectures qui varient (parfois ratées) :** 1587, 1891 principalement. Avec bridge_star ces devraient mieux se trouver.

---

## Structure clé : bridge_star(k, pend)

Graphe bipartite inliné dans solver.py (targeted_graphs + initial_pop) :
- Hub central (0) + k hubs périphériques + k bridges + (k+1)*pend feuilles
- Bipartite → μ = vc (König)
- Viole 1587/1600/1891/2120 (k≥3) et 2252 (k≥5)

```python
G = nx.Graph(); node = k + 1
for i in range(1, k + 1):
    G.add_edge(0, node); G.add_edge(i, node); node += 1
for h in range(k + 1):
    for _ in range(pend):
        G.add_edge(h, node); node += 1
```

---

## Architecture solver.py

| Fonction | Rôle |
|----------|------|
| `compute(G, needed)` | Exact ILP (lent) |
| `compute_fast(G, needed)` | Greedy approx (rapide, filtre) |
| `targeted_graphs(x, y, subgroup)` | Familles ciblées par invariant |
| `initial_pop(conjecture)` | Population initiale |
| `search(conjecture, time_limit, score_fn=None)` | Atlas + SA. score_fn optionnel pour guider le pool |

**Paramètre score_fn dans search() :** injecte la fonction FunSearch pour guider l'ordre d'exploration du pool SA. La détection des contre-exemples reste basée sur violation exacte. Stochastique inchangé.

**ILP_INVARIANTS :** domination_number, total_domination_number, independent_domination_number, independence_number, vertex_cover_number

**Caps :** 40 nœuds (ILP), 100 nœuds (non-ILP)

---

## Architecture FunSearch (Partie 2)

**Sans API, sans LLM. Évolution stochastique de poids.**

- 18 features fixes (violation, diam, Delta, delta, n, m, density, triangles, alpha, tau, mu, td, gamma, randic, td_minus_mu, tau_minus_td, alpha_ratio, deg_spread)
- Chaque candidat = vecteur de poids → généré en code Python via `weights_to_code()`
- Évaluation sur 15 conjectures aléatoires avec `search_with_score_fn` (moteur léger)
- Mutation gaussienne + croisement uniforme + injection aléatoire
- Élitisme : top-6 conservés par itération

**Séparation des rôles :**
- FunSearch évalue avec `search_with_score_fn` (rapide, pour comparer les candidats)
- `apply_best_heuristic` injecte la meilleure fonction dans `solver.search()` (complet, pour le score final)

**`apply_best_heuristic` écrase `results.json` seulement si FunSearch est meilleur.**

---

## Workflow complet

```powershell
# Partie 1 — solver baseline
python src/main.py 60
python src/main.py 90 --retest    # retest uniquement les ratées, plus de temps

# Partie 2 — FunSearch (15-20 min)
python src/funsearch.py 10 6
# Options : python src/funsearch.py 10 6 unsolved   (cible les non-résolues)
#           python src/funsearch.py 10 6 1587 1891  (cible des IDs précis)

# Phase 3 — appliquer meilleure heuristique
python src/apply_best_heuristic.py   # écrase results.json seulement si meilleur

# Vérifier un contre-exemple manuellement
python src/verifier.py 1708 "OdOGEC??G@_N?N??_?GCG"
```

**main.py sauvegarde results.json après chaque conjecture (run complet).**
**--retest sauvegarde une seule fois à la fin.**

---

## Décisions techniques

- **Windows :** pas de SIGALRM → `ThreadPoolExecutor + future.result(timeout=…)`
- **compute_fast() :** greedy pour filtrer avant ILP
- **SA :** T=0.5, cooling=0.997, rechauffe si stale>50
- **Phase atlas :** graphes n≤7 évalués exact
- **FunSearch :** pas de cache, pas de solutions en dur, tout stochastique
- **Pas de triche :** aucun graphe hardcodé dans le pipeline principal
