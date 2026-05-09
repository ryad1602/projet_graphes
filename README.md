# GraphBench — Réfutation automatique de conjectures en théorie des graphes

**Master 1 MIAGE — TD Noté**

---

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
cd src
python main.py 60    # 60 secondes par conjecture (recommandé)
python main.py 10    # 10 secondes par conjecture (test rapide)
```

Les résultats sont sauvegardés dans `results/results.json`.

## Structure

```
graphbench-project/
├── src/
│   ├── conjecture.py   # Parser benchmark + invariants + classes
│   ├── solver.py       # Heuristique complète (Partie 1)
│   ├── funsearch.py    # Architecture FunSearch (Partie 2)
│   └── main.py         # Point d'entrée
├── benchmark/
│   └── benchmark.xlsx
├── results/
├── requirements.txt
└── README.md
```

## Résultats attendus

Avec 60s par conjecture : **~92/100** conjectures réfutées, score total **~1200**.
