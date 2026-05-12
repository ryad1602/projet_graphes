# GraphBench — Réfutation automatique de conjectures en théorie des graphes

**Master 1 MIAGE — TD Noté**

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Structure du projet

```
projet_graphes/
├── benchmark/
│   └── benchmark.xlsx          # 100 conjectures (fourni)
├── src/
│   ├── conjecture.py           # Parser benchmark, invariants, classes de graphes
│   ├── solver.py               # Heuristique principale (SA + ILP + populations ciblées)
│   ├── main.py                 # Runner benchmark complet ou retest
│   ├── funsearch.py            # Optimisation des poids de mutations (FunSearch)
│   ├── apply_best_heuristic.py # Appliquer et comparer les poids FunSearch
│   └── verifier.py             # Vérifier un contre-exemple graph6 manuellement
├── results/
│   ├── results.json            # Résultats du dernier run complet
│   ├── best_mutations.py       # Meilleurs poids FunSearch (généré)
│   └── funsearch_history.json  # Historique FunSearch (généré)
├── requirements.txt
└── README.md
```

---

## Workflow d'exécution

### Étape 1 — Benchmark complet

```bash
py src/main.py 60          # 60s par conjecture (recommandé pour score final)
py src/main.py 10          # 10s par conjecture (test rapide)
py src/main.py 60 5        # Reprendre à partir de la conjecture index 5
```

Si `results/best_mutations.py` existe (généré par FunSearch), les poids sont chargés **automatiquement**.

Résultats sauvegardés dans `results/results.json`.

---

### Étape 2 — Retest des échecs uniquement

```bash
py src/main.py 60 --retest
```

Relance uniquement les conjectures non résolues, fusionne avec les OK existants.

---

### Étape 3 (optionnel) — FunSearch : optimiser les poids de mutations

FunSearch tourne **indéfiniment** (Ctrl+C pour arrêter proprement) et sauvegarde `results/best_mutations.py` à chaque amélioration.

```bash
# Sur les conjectures difficiles + 20 aléatoires (recommandé, ~3 min/itération)
py src/funsearch.py 12 unsolved

# Sur les 100 conjectures (plus lent, ~8 min/itération)
py src/funsearch.py 5

# Personnalisé : N secondes par conjecture
py src/funsearch.py <N>
py src/funsearch.py <N> unsolved
```

> Nécessite une clé API dans `.env` pour le mode LLM (optionnel) :
> ```
> GROQ_API_KEY=gsk_...
> ```
> Sans clé : fonctionne en mode évolutionnaire pur.

---

### Étape 4 (optionnel) — Appliquer et comparer les poids FunSearch

```bash
py src/apply_best_heuristic.py
```

Lance le benchmark complet avec les poids de `results/best_mutations.py` et compare au baseline `results/results.json`.

---

### Outil de débogage — Vérifier un contre-exemple

```bash
py src/verifier.py <conjecture_id> <graph6_string>

# Exemples :
py src/verifier.py 2882 "HCQGOao?"
py src/verifier.py 1708 OdOGEC??G@_N?N??_?GCG
```

Affiche les invariants exacts (ILP), la valeur de la borne et le verdict.

---

## Architecture de l'heuristique

### solver.py — `search(conjecture, time_limit)`

- **Phase 0** : balayage exhaustif de tous les graphes connectés à ≤ 7 sommets (atlas NetworkX)
- **Phase 1** : population initiale de graphes ciblés selon les invariants impliqués, vérification ILP du meilleur
- **Phase 2** : recuit simulé (SA) avec mutations pondérées, seuil ILP adaptatif, hard reset si stagnation (35% du temps restant, max 2 resets)

### Invariants exacts via ILP (PuLP/CBC)

Pour les invariants NP-difficiles : `domination_number`, `total_domination_number`, `independent_domination_number`, `independence_number`, `vertex_cover_number`.
Le SA utilise des approximations greedy rapides ; l'ILP est appelé uniquement sur les candidats prometteurs.

### funsearch.py — Optimisation des poids de mutations

Évolue le dictionnaire `MUTATION_WEIGHTS` (probabilités relatives de sélection des 12 mutations générales, 4 mutations arbre, 6 mutations sans-griffe). Les poids appris sont injectés directement dans le SA de `solver.search()`.

---

## Résultats

| Configuration | Trouvées | Score |
|--------------|----------|-------|
| 60s/conjecture | 100/100 | ~336s |

Score = somme des temps par conjecture trouvée. Échec = pénalité 120s.
