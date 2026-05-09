---
name: GraphBench — contexte projet
description: Architecture, benchmark, état des conjectures, structures contre-exemples découvertes, ce qui reste à faire
type: project
---

## Objectif

Trouver des contre-exemples à 100 conjectures (toutes REJECTED = fausses). Score = somme des temps d'exécution → minimiser.

**How to apply:** Orienter sur vitesse de convergence + qualité population initiale. Toutes les conjectures ont un contre-exemple connu dans le benchmark (Counter X, Counter Y, Counter example order).

---

## Structure du projet

```
projet_graphes/
├── benchmark/benchmark.xlsx   ← 100 conjectures (pandas)
├── src/
│   ├── main.py        ← orchestrateur, ThreadPoolExecutor (Windows = pas de SIGALRM)
│   ├── conjecture.py  ← parser + invariants (ne pas modifier)
│   ├── solver.py      ← moteur heuristique (LE fichier clé)
│   └── verifier.py    ← CLI: python verifier.py <id> <g6>  ← nouveau, fonctionnel
└── results/results.json
```

---

## État des 11 conjectures total_domination difficiles

### RÉSOLUES (contre-exemple à ajouter dans solver.py)

| ID | Condition violation | Structure gagnante | Score |
|----|--------------------|--------------------|-------|
| **1566** | td > alpha+1 | `bridge_star(k=6,pend=1)` OU graphe 1708 | via 1708 graph6 |
| **1708** | td > alpha+1 | graphe6 `OdOGEC??G@_N?N??_?GCG` (n=16) | 1.0 |
| **1587** | matching < (2/3)td - 1/3 | `bridge_star(k≥3,pend=1)` (bipartite, μ=vc) | 0.333+ |
| **1600** | vc < (2/3)td - 1/3 | `bridge_star(k≥3,pend=1)` | 0.333+ |
| **1891** | td > 1.5*vc + 0.5 | `bridge_star(k≥3,pend=1)` | 0.5+ |
| **2120** | td > 1.5*mu + 0.5 | `bridge_star(k≥3,pend=1)` (idem 1891 car bipartite) | 0.5+ |
| **2252** | td > f(randic) cubique | `bridge_star(k≥5,pend=1)` (k=5: score=1.011) | 1.011+ |
| **2051** | vc < 3/4 + 1/4*idom | Double star asymétrique S_{k1,k2}, k1≥5, k2>k1 | 0.25+ |

### ENCORE BLOQUÉES (pas de contre-exemple trouvé)

| ID | Condition | Subgroup | Taille cible | Note |
|----|-----------|----------|--------------|------|
| **6574** | alpha > td+1 | claw_free+connected | n=18, td=6, alpha=8 | Impossible sur line graphs |
| **6582** | matching < td-1 | claw_free+connected | n=17, td=10, mu=8 | Impossible sur line graphs |
| **6903** | td > matching+1 | claw_free+connected | n=19, td=11, mu=9 | Impossible sur line graphs |

**Pourquoi impossible sur line graphs :** Pour tout L(G) connexe, γ_t(L(G)) ≤ μ(L(G)). Donc les contre-exemples pour 6582/6903 doivent être des graphes claw_free qui NE SONT PAS des line graphs.

**Pistes claw_free non explorées :**
- `C_n^2` (carré du cycle) : claw_free prouvé, non line graph pour n≥6
- Graphes circulants `C(n, {1,2,...,k})`
- `K_k + feuilles` (attention : seulement K_3 ou K_4, sinon degré élevé → claw possible)
- Script `src/search_clawfree.py` créé mais pas encore exécuté

---

## Structure clé : bridge_star(k, pend)

La découverte principale de la session. Graphe bipartite :
- 1 hub central (vertex 0) + k hubs périphériques (vertices 1..k)
- k "bridge" leaves : chaque bridge adj au hub central ET à un hub périphérique
- (k+1)*pend feuilles exclusives (une par hub)
- **Aucune edge entre hubs** → bipartite → μ = vc (König's theorem)

```
Invariants analytiques :
  n   = (k+1) + k + (k+1)*pend = 3k + 1 + (k+1)*pend
  vc  = k+1,  alpha = n - vc
  td  = 2k+1  (4 hubs en S + k bridges en S)
  mu  = vc = k+1  (bipartite)
  Randic ≈ dépend de la structure

Scores (pend=1) :
  1891 = (2k+1) - (1.5*(k+1)+0.5) = 0.5*(k-2)
  1600 = (2k+1 - 1/3) - (k+1) * 2/3 = (k-2)/3
  k=3 : 1891=0.5, 1600=0.333
  k=4 : 1891=1.0, 1600=0.667
  k=5 : 1891=1.5, 1600=1.0  AND  2252=1.011  ← aussi 2252!
  k=6 : 1891=2.0, 2252=1.803
```

**Code Python :**
```python
def bridge_star(k, pend=1):
    G = nx.Graph()
    node = k + 1
    for i in range(1, k + 1):          # k bridges
        G.add_edge(0, node); G.add_edge(i, node); node += 1
    for h in range(k + 1):             # (k+1)*pend exclusive leaves
        for _ in range(pend):
            G.add_edge(h, node); node += 1
    return G
```

---

## Ce qui reste à faire dans solver.py

### 1. Ajouter bridge_star à `targeted_graphs` (section total_domination_number)

```python
if 'total_domination_number' in invs:
    # bridge_star: violates 1587/1600/1891/2120 (k>=3) and 2252 (k>=5)
    for k in range(2, 9):
        for pend in range(1, 4):
            G = nx.Graph(); node = k+1
            for i in range(1, k+1):
                G.add_edge(0, node); G.add_edge(i, node); node += 1
            for h in range(k+1):
                for _ in range(pend):
                    G.add_edge(h, node); node += 1
            graphs.append(G)
```

### 2. Ajouter dans `initial_pop` (branche else/connected)

```python
# bridge_star family (violates 1587/1600/1891/2120/2252)
for k in range(2, 9):
    for pend in range(1, 4):
        G = nx.Graph(); node = k+1
        for i in range(1, k+1):
            G.add_edge(0, node); G.add_edge(i, node); node += 1
        for h in range(k+1):
            for _ in range(pend):
                G.add_edge(h, node); node += 1
        pop.append(G)

# Known 1708/1566 counterexample (n=16, td=8, alpha=6, score=1.0)
pop.append(nx.from_graph6_bytes(b'OdOGEC??G@_N?N??_?GCG'))
```

### 3. Double stars asymétriques pour 2051

Déjà présents dans `targeted_graphs` et `initial_pop` (ajoutés session précédente). Vérifier qu'ils couvrent k1≥5.

### 4. Claw_free : exécuter search_clawfree.py et analyser les résultats

```powershell
cd src; python search_clawfree.py
```

---

## Architecture solver.py (résumé)

| Fonction | Rôle |
|----------|------|
| `compute(G, needed)` | Exact ILP (lent) |
| `compute_fast(G, needed)` | Greedy approx (rapide, filtre) |
| `_fast_total_dom(G)` | Greedy pour td (ajouté, remplace 2*mu) |
| `targeted_graphs(x, y, subgroup)` | Familles ciblées par invariant |
| `initial_pop(conjecture)` | Population initiale (cap 2*size) |
| `search(conjecture, time_limit)` | Phase atlas (n≤7 exact) + SA |

**ILP_INVARIANTS** : `domination_number`, `total_domination_number`, `independent_domination_number`, `independence_number`, `vertex_cover_number`

**Caps de taille** : 40 nœuds (ILP), 100 nœuds (non-ILP)

---

## Commandes utiles

```powershell
# Lancer depuis la racine
python src/main.py 60       # 60s par conjecture

# Vérifier un contre-exemple
python src/verifier.py 1708 "OdOGEC??G@_N?N??_?GCG"
python src/verifier.py 1891 "J?eA__oGA??"   # bridge_star(k=3)

# Chercher des structures claw_free
python src/search_clawfree.py
```

---

## Décisions techniques

- **Windows** : pas de SIGALRM → `ThreadPoolExecutor + future.result(timeout=…)`
- **compute_fast()** : greedy pour filtrer 90% des candidats avant ILP
- **Recuit simulé** : T=0.5, cooling=0.997, rechauffe si stale>50
- **Phase atlas** : graphes n≤7 évalués exact (ILP instantané à cette taille)
