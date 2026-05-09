import json
import networkx as nx
from conjecture import load_benchmark, to_graph6

def verify_results(json_path, benchmark_path):
    # 1. Charger les résultats et le benchmark
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    all_conjectures = load_benchmark(benchmark_path)
    # Créer un dictionnaire pour retrouver vite une conjecture par son ID
    conj_dict = {str(c.id): c for c in all_conjectures}

    print(f"=== Vérification de {len(data['results'])} résultats ===\n")

    for res in data['results']:
        if res['status'] != "OK":
            continue

        c_id = str(res['conjecture_id'])
        g6_str = res['g6']
        conj = conj_dict[c_id]

        # 2. Reconstruire le graphe
        G = nx.from_graph6_string(g6_str)

        # 3. Recalculer les invariants (via ton fichier conjecture.py)
        # On vérifie si la condition de violation est toujours vraie
        val_x = conj.calculate_invariant(G, conj.x_name)
        val_y = conj.calculate_invariant(G, conj.y_name)
        
        # Calculer la violation selon le signe de la conjecture
        if conj.sign == "<=":
            violation = val_y - conj.evaluate_expression(val_x)
        else:
            violation = conj.evaluate_expression(val_x) - val_y

        if violation > 1e-9:
            print(f"✅ Conjecture {c_id} : VALIDÉE (Violation réelle: {violation:.4f})")
        else:
            print(f"❌ Conjecture {c_id} : ERREUR ! Pas de violation réelle.")

# Lancer la vérification
verify_results('../results/results.json', '../benchmark/benchmark.xlsx')