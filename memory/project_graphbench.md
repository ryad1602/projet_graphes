---
name: GraphBench — contexte projet
description: Architecture, état du code, conjectures difficiles et décisions de design pour le projet GraphBench
type: project
---

## Objectif

Trouver des contre-exemples à 100 conjectures mathématiques sur des graphes (toutes de statut REJECTED = falsifiables).
Score = somme des temps d'exécution → minimiser le temps total.

**Why:** Compétition / travail académique. Toutes les conjectures sont déjà connues comme fausses, il faut juste retrouver les contre-exemples.

**How to apply:** Orienter les efforts sur la vitesse de convergence et la qualité de la population initiale.

---

## Structure du projet

```
projet_graphes/
├── benchmark/benchmark.xlsx   ← 100 conjectures (lire avec pandas)
├── src/
│   ├── main.py        ← orchestrateur, timeout via concurrent.futures
│   ├── conjecture.py  ← parser + calcul de tous les invariants
│   └── solver.py      ← moteur de recherche heuristique (LE fichier clé)
└── results/results.json       ← créé automatiquement à la fin
```

---

## Analyse du benchmark (100 conjectures)

### Distribution des subgroups
| Subgroup | Nombre |
|----------|--------|
| `['connected']` | 44 |
| `['connected', 'tree']` | 6 |
| `['claw_free', 'connected']` | 50 |

### Tailles des contre-exemples (observées dans le CSV)
- Min : 9 nœuds, Max : 90 nœuds (conjecture 886 outlier)
- Médiane : 17 nœuds
- ILP : n=14-31, Spectral/distance : n=9-25

### Conjectures ILP (43/100) — les plus lentes
Impliquent au moins un de : `domination_number`, `total_domination_number`,
`independence_number`, `vertex_cover_number`, `independent_domination_number`

### Les 8 conjectures les plus dures (>60s dans benchmark original)
| ID | Paire d'invariants | Temps original |
|----|-------------------|----------------|
| 2120 | matching / total_domination | 185s |
| 6574 | total_domination / independence (claw-free) | 166s |
| 6582 | total_domination / matching (claw-free) | 142s |
| 2041 | independent_domination / matching | 131s |
| 1600 | total_domination / vertex_cover | 156s |
| 6903 | matching / total_domination (claw-free) | 82s |
| 7703 | largest_distance_eigenvalue / second_smallest_laplace (claw-free) | 78s |
| 6011 | radius / remoteness (claw-free) | 71s |

### Conjecture spéciale
- **886** : `maximum_degree` vs `triangle_number`, contre-exemple à **n=90**, max_degree=6, triangles=39. Nécessite des graphes k-réguliers grands (k=5-8, n=40-90).

---

## Architecture de solver.py

### Fonctions clés

| Fonction | Rôle |
|----------|------|
| `compute(G, needed)` | Calcul exact avec ILP (lent pour dom/indep) |
| `compute_fast(G, needed)` | Approximation greedy (rapide, filtre pré-ILP) |
| `targeted_graphs(x, y, subgroup)` | Génère des familles ciblées selon les invariants |
| `initial_pop(conjecture)` | Population initiale = targeted + génériques, cap 25 |
| `search(conjecture, time_limit)` | Moteur principal : 2 phases + recuit simulé |
| `_needs_ilp(needed)` | Détecte si la conjecture nécessite ILP |

### Logique de search()

```
Phase 1 (15% du temps max) :
  - 25 graphes de initial_pop()
  - Évalués avec compute_fast() uniquement
  - Si ILP : valide le meilleur avec compute() exact

Phase 2 (boucle de mutation, 85% du temps) :
  - Sélection tournament (k=3) depuis pool
  - 1-3 mutations aléatoires + repair()
  - Cap taille : 40 nœuds (ILP) ou 100 nœuds (non-ILP)
  - Évaluation 2 phases : fast d'abord, ILP si score_fast > best - 1.0
  - Recuit simulé : T=0.5, cooling=0.997, rechauffe si stale>50
  - Stagnation : mutations depuis meilleur connu + 3 graphes aléatoires
```

### Caps de taille (IMPORTANT)
```python
max_nn = 40 if use_ilp else 100
```
- ILP : 40 (contre-exemples réels en n=14-31)
- Non-ILP : 100 (conjecture 886 a besoin de n=90)

---

## Décisions techniques importantes

### Pourquoi concurrent.futures dans main.py ?
`signal.SIGALRM` n'existe pas sur Windows. `ThreadPoolExecutor` avec `future.result(timeout=hard_limit)` assure que chaque conjecture est tuée après `int(time_limit * 1.3) + 5` secondes.

### Pourquoi compute_fast() ?
L'ILP (PuLP/CBC) sur un graphe à 30 nœuds peut prendre 5-10s. Dans la boucle de mutation, on évalue des centaines de candidats → sans filtrage, tout le temps est mangé par l'ILP. `compute_fast()` utilise des approximations greedy (domination = greedy, independence = MIS greedy) pour filtrer 90% des candidats sans payer le coût ILP.

### Pourquoi le recuit simulé ?
La boucle originale était purement greedy (accepte seulement les améliorations). Elle stagnait rapidement dans des optima locaux, surtout pour les invariants ILP qui ont des paysages de fitness non-lisses.

---

## Familles de graphes par invariant (targeted_graphs)

| Invariant | Graphes cibles |
|-----------|---------------|
| `maximum_degree + triangle_number` | k-réguliers n=20-90 (k=4-8), K_n |
| `second_smallest_laplace_eigenvalue` | Barbells K_k—edge—K_k (λ2 très petit), lollipops |
| `largest_distance_eigenvalue` | Cycles, line-paths compacts (n=6-15) |
| `first_zagreb + second_zagreb` | K_n, k-réguliers denses (k=4-10), barbells |
| `domination/total_domination` | Chemins, cycles, grilles k×k, bipartis |
| `independence/vertex_cover/ind_dom` | Bipartis complets, cycles pairs, Erdos-Renyi creux |
| `matching_number` | Bipartis complets, graphes 3-réguliers |
| `clique_number` | K_n, Erdos-Renyi dense (p=0.6-0.9) |
| `diameter/radius` | Chemins, cycles, lollipops, caterpillars |
| `proximity/remoteness` | Étoiles (haute proximity), chemins (basse proximity) |
| `average_degree + maximum_degree` | Erdos-Renyi variés, barbells, line graphs K_n |
| `triangle + largest_eigenvalue` | K_n, barbells, line graphs K_n |

---

## Ce qui reste à améliorer (pistes)

1. **Conjectures >60s** : Envisager de partager le temps différemment (donner 120s aux 8 conjectures dures, 30s aux faciles) — nécessite une détection préalable de difficulté.
2. **Réutiliser les contre-exemples** : Un graphe qui viole la conjecture A (partageant des invariants avec B) pourrait être essayé sur B en premier.
3. **ILP plus rapide** : Utiliser des solvers alternatifs (HiGHS via scipy) ou renforcer les contraintes LP pour CBC.
4. **Taille cible adaptative** : Pour les ILP, se concentrer sur n=14-22 plutôt que 14-40, car la médiane des contre-exemples est 17.

---

## Commandes utiles

```powershell
# Lancer depuis la racine du projet
python src/main.py 60      # 60s par conjecture
python src/main.py 30      # 30s (test rapide)

# Résultats dans results/results.json
```
