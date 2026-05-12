---
name: GraphBench — contexte projet
description: Architecture complète, résultats, décisions techniques, workflow — pour rédaction rapport
type: project
---

## Objectif

Trouver des contre-exemples à 100 conjectures mathématiques sur les graphes (toutes REJECTED = fausses).
Score = somme des temps d'exécution (secondes) → **minimiser**.
Pénalité FAIL/TO = 120 pts par conjecture non résolue.

---

## Résultats obtenus

| Run | Score | Trouvées |
|-----|-------|----------|
| Solver baseline (main.py, 60s) | **327.01** | **100/100** |
| FunSearch optimisé (apply_best_heuristic) | 592.47 | 98/100 |

Le solver baseline est meilleur — `apply_best_heuristic.py` écrase `results.json` seulement si FunSearch améliore le score (ce n'était pas le cas ici).

**Score < 100 est théoriquement possible** (moyenne < 1s/conjecture) mais difficile : certaines conjectures nécessitent SA + ILP qui prennent plusieurs secondes. 327 avec 100/100 est un excellent résultat.

---

## Structure du projet

```
projet_graphes/
├── benchmark/benchmark.xlsx      ← 100 conjectures (pandas)
├── src/
│   ├── main.py                   ← orchestrateur ThreadPoolExecutor (216 lignes)
│   ├── conjecture.py             ← parser + invariants — NE PAS MODIFIER (328 lignes)
│   ├── solver.py                 ← moteur heuristique principal (1238 lignes)
│   ├── funsearch.py              ← Partie 2 : évolution heuristique (682 lignes)
│   ├── apply_best_heuristic.py   ← Phase 3 : relancer avec meilleure fonction (215 lignes)
│   └── verifier.py               ← CLI: python verifier.py <id> <g6>
├── results/
│   ├── results.json              ← résultats solver baseline (327.01 — 100/100)
│   ├── results_optimized.json    ← résultats avec heuristique FunSearch (592.47 — 98/100)
│   ├── funsearch_history.json    ← historique des itérations FunSearch
│   └── best_heuristic.py         ← meilleure fonction de score trouvée
├── .env                          ← GROQ_API_KEY (gitignored)
└── .gitignore                    ← ignore .env, __pycache__, *.pyc
```

---

## Pipeline complet (Partie 1 + Partie 2 + Phase 3)

### Partie 1 — Solver baseline

```powershell
python src/main.py 60                  # 60s par conjecture, sauvegarde après chaque
python src/main.py 90 --retest         # retest uniquement les ratées avec plus de temps
```

`main.py` utilise `ThreadPoolExecutor` (contournement de l'absence de SIGALRM sur Windows).
Sauvegarde `results/results.json` après chaque conjecture (run complet).
`--retest` sauvegarde une seule fois à la fin.

### Partie 2 — FunSearch (évolution de la fonction de score)

```powershell
python src/funsearch.py 10 6               # 10 itérations, 6s par évaluation
python src/funsearch.py 10 6 unsolved      # cible les conjectures ratées/lentes
python src/funsearch.py 10 6 1587 1891    # cible des IDs précis
```

Sauvegarde :
- `results/best_heuristic.py` — meilleure fonction Python trouvée
- `results/funsearch_history.json` — historique des coûts par itération

### Phase 3 — Appliquer la meilleure heuristique

```powershell
python src/apply_best_heuristic.py    # écrase results.json seulement si FunSearch est meilleur
```

### Vérification manuelle

```powershell
python src/verifier.py 1708 "OdOGEC??G@_N?N??_?GCG"
```

---

## Architecture solver.py (1238 lignes)

| Bloc | Lignes approx. | Rôle |
|------|----------------|------|
| ILP functions (5 fonctions) | ~80 | Résolution exacte par programmation linéaire entière |
| `compute(G, needed)` | ~90 | Calcul exact des invariants (lent, pour vérification finale) |
| `compute_fast(G, needed)` + helpers | ~130 | Approximation greedy (filtre rapide avant ILP) |
| `targeted_graphs(x, y, subgroup)` | ~290 | Familles de graphes ciblées par invariant |
| `initial_pop(conjecture)` | ~230 | Population initiale SA (utilise targeted_graphs) |
| Mutations (12 fonctions) + `repair()` | ~120 | Opérateurs de modification de graphes |
| `search(conjecture, time_limit, score_fn)` | ~215 | Atlas + SA avec hard reset |
| `verify_exact()` | ~10 | Vérification finale exacte |

**ILP_INVARIANTS** (calcul exact via PuLP) :
domination_number, total_domination_number, independent_domination_number, independence_number, vertex_cover_number

**Caps :** 40 nœuds (ILP), 100 nœuds (non-ILP)

**SA params :** T=0.5, cooling=0.997, rechauffe si stale>50. Hard reset jusqu'à 2 fois par run si stagnation.

---

## Graphe bridge_star(k, pend) — innovation clé

Famille de graphes bipartites injectée dans `targeted_graphs()` et `initial_pop()` pour cibler les conjectures 1587/1600/1891/2120/2252.

Structure : hub central (0) + k hubs périphériques + k bridges + (k+1)*pend feuilles.
Bipartite → μ = vc (König) → viole les conjectures liant matching_number et vertex_cover.

```python
G = nx.Graph(); node = k + 1
for i in range(1, k + 1):
    G.add_edge(0, node); G.add_edge(i, node); node += 1
for h in range(k + 1):
    for _ in range(pend):
        G.add_edge(h, node); node += 1
```

---

## Architecture FunSearch (funsearch.py, 682 lignes)

### Principe

Évolution automatique de la fonction `heuristic_score(G, invariants, conjecture)` qui guide l'ordre d'exploration du SA. La détection des contre-exemples reste basée sur la violation exacte — le score_fn ne triche pas.

### Deux modes

**Mode LLM (GROQ_API_KEY définie dans .env) :**
- Backend : Groq (gratuit, llama-3.1-8b-instant) en priorité, Anthropic en fallback
- Chaque itération : 3 variantes générées par le LLM + 1 évolutionnaire pour la diversité
- Si toutes les variantes LLM échouent → bascule automatique en mode évolutionnaire

**Mode évolutionnaire (sans API) :**
- 18 features (violation, diam, Delta, delta, n, m, density, triangles, alpha, tau, mu, td, gamma, randic, + 4 dérivées)
- Chaque candidat = vecteur de poids → converti en code Python via `weights_to_code()`
- Mutation gaussienne + croisement uniforme + injection aléatoire
- Élitisme : top-6 conservés par itération, population_size=6

### Détection du mode (affichage)

```
=== FunSearch — mode LLM (GROQ) ===        # si GROQ_API_KEY dans .env
=== FunSearch — mode évolutionnaire (sans API) ===   # sinon
```

### Clé API

Stockée dans `.env` à la racine (gitignored). Chargée automatiquement au démarrage du module (lignes 22-32 de funsearch.py), jamais hardcodée.

```
GROQ_API_KEY=gsk_...
```

### Séparation des rôles

- `search_with_score_fn()` — moteur léger pour comparer les candidats (FunSearch interne)
- `solver.search()` avec `score_fn=` — moteur complet pour le score final (apply_best_heuristic)

### Prompt envoyé au LLM

**System prompt** (`_LLM_SYSTEM`) : décrit la signature exacte de la fonction, les 18 variables disponibles, les règles (pas d'import, violation ≥ 1.0, scalaire en retour).

**User prompt** (`_LLM_PROMPT_TPL`) : fournit le code de la meilleure fonction actuelle + son coût, demande une variante améliorée.

Premier appel : la fonction de base (`BASE_WEIGHTS` — poids manuels sur violation/diam/Delta/density/triangles) est envoyée comme exemple.

### Évaluation

`evaluate_score_function(code, sample, time_per_conj)` évalue sur un sous-ensemble de conjectures (15 max). Coût = somme des temps trouvés + 2×time_per_conj pour les ratées.

---

## Décisions techniques clés

| Problème | Solution retenue |
|----------|-----------------|
| Windows sans SIGALRM | `ThreadPoolExecutor + future.result(timeout=…)` dans main.py |
| API Anthropic payante | Groq (gratuit, llama-3.1-8b-instant) via package `groq` |
| Clé API sécurisée | `.env` gitignored, chargé à l'import de funsearch.py |
| Groq unavailable mid-run | Fallback évolutionnaire automatique si llm_ok==0 |
| compute() trop lent | compute_fast() greedy filtre avant ILP |
| Stagnation SA | Hard reset (jusqu'à 2×), réchauffage si stale>50 |
| Population structure FunSearch | 4-tuple (cost, rate, weights, code_str) |

---

## État des conjectures

Conjectures parfois difficiles (variance stochastique) : **1587, 1891**.
Avec bridge_star dans targeted_graphs + initial_pop, ces conjectures sont beaucoup mieux couvertes.

Résultats typiques sur run complet : **94–100/100 trouvées, score 327–1200** selon seed et time_limit.
Meilleur résultat obtenu : **327.01 — 100/100**.
