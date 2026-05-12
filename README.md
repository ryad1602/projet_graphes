# GraphBench — Réfutation automatique de conjectures en théorie des graphes

**Master 1 MIAGE — TD Noté**

---

## Installation

```bash
pip install -r requirements.txt
```

Les dépendances LLM (`groq`, `anthropic`) sont optionnelles — FunSearch fonctionne sans elles en mode évolutionnaire pur.

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
│   ├── apply_best_heuristic.py # Appliquer les poids FunSearch et comparer au baseline
│   └── verifier.py             # Vérifier un contre-exemple graph6 manuellement
├── results/
│   ├── results.json            # Résultats du dernier run complet (100/100)
│   ├── best_mutations.py       # Meilleurs poids trouvés par FunSearch
│   └── funsearch_history.json  # Historique des améliorations FunSearch
├── requirements.txt
└── README.md
```

---

## Workflow d'exécution

### Étape 1 — Benchmark complet (solver seul)

```bash
python src/main.py 60          # 60s par conjecture (recommandé pour score final)
python src/main.py 10          # 10s par conjecture (test rapide)
python src/main.py 60 5        # Reprendre à partir de la conjecture index 5
```

Résultats sauvegardés dans `results/results.json`.

---

### Étape 2 — Retest des échecs uniquement

```bash
python src/main.py 60 --retest
```

Relance uniquement les conjectures non résolues, fusionne avec les OK existants.

---

### Étape 3 — FunSearch : optimiser les poids de mutations

FunSearch fait évoluer les probabilités de sélection des mutations du recuit simulé en s'appuyant sur un LLM (GROQ) et des opérateurs évolutionnaires. Il tourne **indéfiniment** (Ctrl+C pour arrêter proprement) et sauvegarde `results/best_mutations.py` uniquement quand il trouve mieux que la meilleure solution connue — même entre plusieurs relances.

Créer un fichier `.env` à la racine avec la clé GROQ :
```
GROQ_API_KEY=gsk_...
```

Puis lancer :
```bash
python src/funsearch.py 5      # 5s par conjecture, toutes les 100 conjectures
python src/funsearch.py <N>    # N secondes par conjecture
```

À chaque itération, le LLM analyse les statistiques de performance par classe de graphes (arbres, sans-griffe, général) et propose de nouveaux poids motivés. Les opérateurs évolutionnaires (mutation, crossover) complètent l'exploration. Le meilleur résultat est sauvegardé dans `results/best_mutations.py`.

---

### Étape 4 — Appliquer les poids FunSearch et comparer

```bash
python src/apply_best_heuristic.py
```

Lance le benchmark complet avec les poids de `results/best_mutations.py` et compare au baseline `results/results.json`. Le solver (`main.py`) et FunSearch sont **indépendants** — les poids optimisés ne sont utilisés que via ce script.

---

### Outil de débogage — Vérifier un contre-exemple

```bash
python src/verifier.py <conjecture_id> <graph6_string>

# Exemples :
python src/verifier.py 2882 "HCQGOao?"
python src/verifier.py 1708 OdOGEC??G@_N?N??_?GCG
```

Affiche les invariants exacts (ILP), la valeur de la borne et le verdict.

---

## Architecture de l'heuristique

### solver.py — `search(conjecture, time_limit)`

- **Phase 0** : balayage exhaustif de tous les graphes connectés à ≤ 7 sommets (atlas NetworkX, ~830 graphes). Réfute ~33/100 conjectures en quelques secondes.
- **Phase 1** : population initiale de graphes ciblés selon les invariants impliqués (étoiles, chemins, cycles, graphes bipartis, double-étoiles...), vérification ILP du meilleur candidat.
- **Phase 2** : recuit simulé (SA) avec 12 mutations pondérées (général), 4 (arbres), 6 (sans-griffe). Seuil ILP adaptatif, hard reset si stagnation (35% du temps restant, max 2 resets).

### Invariants exacts via ILP (PuLP/CBC)

Pour les invariants NP-difficiles : `domination_number`, `total_domination_number`, `independent_domination_number`, `independence_number`, `vertex_cover_number`.
Le SA utilise des approximations greedy rapides ; l'ILP est appelé uniquement sur les candidats prometteurs pour valider le contre-exemple.

### funsearch.py — Méta-optimisation des poids de mutations

Optimise le dictionnaire `MUTATION_WEIGHTS` (probabilités relatives de sélection des mutations) par recherche évolutionnaire guidée par LLM :

1. **Évaluation** : chaque candidat est testé sur les 100 conjectures, le score est la somme des temps de recherche (même métrique que le benchmark).
2. **Génération** : le LLM (si disponible) analyse les statistiques par classe de graphes et propose de nouveaux poids motivés par la théorie. L'évolution (mutation, crossover) complète la diversité.
3. **Sélection** : seuls les poids strictement meilleurs que le meilleur connu sont sauvegardés, y compris entre plusieurs relances.

---

## Résultats

| Configuration | Trouvées | Score (lower is better) |
|---|---|---|
| Solver seul — 60s/conjecture | 100/100 | ~336s |
| Solver + poids FunSearch | 100/100 | en cours |

Score = somme des temps par conjecture trouvée. Échec = pénalité 120s.
