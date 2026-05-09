"""
FunSearch – Partie 2 : évolution automatique de la fonction de score.
Dissociée de la Partie 1 (solver.py). Aucune API externe requise.

Principe :
  1. Une population de fonctions heuristic_score est maintenue.
  2. Chaque candidat est encodé comme un vecteur de poids sur une base de features.
  3. Mutation + croisement génèrent de nouveaux candidats.
  4. Les meilleurs sont conservés (élitisme).
  5. Le meilleur code Python final est exporté.
"""
import random
import time
import json
import os
import sys
import networkx as nx

# Partie 1 (moteur de recherche) — importée mais non modifiée
from conjecture import load_benchmark, check_graph_class
from solver import (
    compute_fast, initial_pop, ALL_MUTS, CF_MUTS, TREE_MUTS,
    repair, rnd_claw_free, rnd_connected, rnd_tree,
)


# ──────────────────────────────────────────────────────────────
# Base de features : (nom_variable_locale, clé_invariant_ou_None)
# L'index 0 est toujours 'violation', son poids est contraint ≥ 1.0
# Les features dérivées (None) sont calculées dans le corps généré.
# ──────────────────────────────────────────────────────────────
FEATURES = [
    ('violation',    None),              # conjecture.violation(invariants)
    ('diam',         'diameter'),
    ('Delta',        'maximum_degree'),
    ('delta',        'minimum_degree'),
    ('n',            'order'),
    ('m',            'size'),
    ('density',      'density'),
    ('triangles',    'triangle_number'),
    ('alpha',        'independence_number'),
    ('tau',          'vertex_cover_number'),
    ('mu',           'matching_number'),
    ('td',           'total_domination_number'),
    ('gamma',        'domination_number'),
    ('randic',       'randic_index'),
    ('td_minus_mu',  None),              # td - mu
    ('tau_minus_td', None),              # tau - td
    ('alpha_ratio',  None),              # alpha / n
    ('deg_spread',   None),              # Delta - delta
]

N_FEATURES = len(FEATURES)

# Poids de départ correspondant à la fonction exemple du sujet
BASE_WEIGHTS = [
    10.0,   # violation
     0.3,   # diam
     0.2,   # Delta
     0.0,   # delta
    -0.05,  # n
     0.0,   # m
    -0.2,   # density
     0.1,   # triangles
     0.0,   # alpha
     0.0,   # tau
     0.0,   # mu
     0.0,   # td
     0.0,   # gamma
     0.0,   # randic
     0.0,   # td_minus_mu
     0.0,   # tau_minus_td
     0.0,   # alpha_ratio
     0.0,   # deg_spread
]
assert len(BASE_WEIGHTS) == N_FEATURES


# ──────────────────────────────────────────────────────────────
# Génération de code : vecteur de poids → fonction Python
# ──────────────────────────────────────────────────────────────

def weights_to_code(weights):
    """
    Convertit un vecteur de poids en une fonction heuristic_score
    respectant exactement la forme imposée par le sujet.
    """
    lines = [
        'def heuristic_score(G, invariants, conjecture):',
        '    """',
        '    G : graphe NetworkX',
        '    invariants : dictionnaire des invariants calcules',
        '    conjecture : objet decrivant la conjecture',
        '    retourne un score numerique a maximiser',
        '    """',
        '    violation = conjecture.violation(invariants)',
        '    n         = invariants.get("order", 0)',
        '    m         = invariants.get("size", 0)',
        '    delta     = invariants.get("minimum_degree", 0)',
        '    Delta     = invariants.get("maximum_degree", 0)',
        '    diam      = invariants.get("diameter", 0)',
        '    gamma     = invariants.get("domination_number", 0)',
        '    alpha     = invariants.get("independence_number", 0)',
        '    tau       = invariants.get("vertex_cover_number", 0)',
        '    triangles = invariants.get("triangle_number", 0)',
        '    mu        = invariants.get("matching_number", 0)',
        '    td        = invariants.get("total_domination_number", 0)',
        '    density   = invariants.get("density", 0)',
        '    randic    = invariants.get("randic_index", 0)',
        '    td_minus_mu  = td - mu',
        '    tau_minus_td = tau - td',
        '    alpha_ratio  = alpha / n if n > 0 else 0',
        '    deg_spread   = Delta - delta',
        '    return (',
    ]
    terms = []
    for i, (name, _) in enumerate(FEATURES):
        w = weights[i]
        if abs(w) > 1e-6:
            terms.append(f'        {w:+.4f} * {name}')
    lines.append('\n'.join(terms) if terms else '        0.0')
    lines.append('    )')
    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────
# Opérateurs évolutionnaires (remplacent le LLM)
# ──────────────────────────────────────────────────────────────

def random_weights():
    """Vecteur aléatoire. Violation toujours dominante (≥ 5.0)."""
    w = [random.uniform(-0.5, 0.5) for _ in range(N_FEATURES)]
    w[0] = random.uniform(5.0, 15.0)
    return w


def mutate_weights(weights, sigma=0.3):
    """
    Mutation gaussienne + toggles aléatoires.
    Simule ce qu'un LLM ferait : petites variations + exploration de nouvelles features.
    """
    w = list(weights)
    for i in range(N_FEATURES):
        if random.random() < 0.6:
            w[i] += random.gauss(0, sigma)
        if random.random() < 0.1:    # éteindre une feature
            w[i] = 0.0
        if random.random() < 0.1:    # activer une feature dormante
            w[i] = random.gauss(0, 0.3)
    w[0] = max(1.0, w[0])            # violation toujours positive
    return w


def crossover_weights(w1, w2):
    """Croisement uniforme entre deux parents."""
    child = [w1[i] if random.random() < 0.5 else w2[i] for i in range(N_FEATURES)]
    child[0] = max(1.0, child[0])
    return child


# ──────────────────────────────────────────────────────────────
# Moteur de recherche guidé par la fonction de score
# (structure identique à solver.py mais pilotée par score_fn externe)
# ──────────────────────────────────────────────────────────────

_EXTRA_NEEDED = {
    'order', 'size', 'minimum_degree', 'maximum_degree',
    'diameter', 'density', 'domination_number', 'total_domination_number',
    'independence_number', 'vertex_cover_number', 'matching_number',
    'triangle_number', 'randic_index',
}


def search_with_score_fn(conjecture, score_fn, time_limit=8):
    """
    Recherche de contre-exemple guidée par score_fn.
    Retourne (G, inv, violation, elapsed).
    """
    start = time.time()
    subgroup = conjecture.subgroup
    needed = conjecture.required_invariant_names() | _EXTRA_NEEDED

    if 'claw_free' in subgroup:
        muts = CF_MUTS
    elif 'tree' in subgroup:
        muts = TREE_MUTS
    else:
        muts = ALL_MUTS

    population = initial_pop(conjecture, size=20)
    best_graph, best_violation, best_inv = None, float('-inf'), None
    pool = []

    # Phase initiale : évaluer la population de départ
    for G in population:
        if time.time() - start > time_limit:
            break
        if not check_graph_class(G, subgroup):
            continue
        try:
            inv = compute_fast(G, needed)
            violation = conjecture.violation(inv)
            score = score_fn(G, inv, conjecture)
            pool.append((score, G, inv, violation))
            if violation > best_violation:
                best_violation, best_graph, best_inv = violation, G, inv
                if violation > 1e-9:
                    return G, inv, violation, time.time() - start
        except Exception:
            pass

    pool.sort(key=lambda x: x[0], reverse=True)
    pool = pool[:20]

    # Phase principale : mutation guidée par le score
    while time.time() - start < time_limit:
        if pool:
            _, G, _, _ = pool[random.randint(0, min(4, len(pool) - 1))]
        else:
            n = random.randint(5, 30)
            try:
                if 'claw_free' in subgroup:
                    G = rnd_claw_free(n)
                elif 'tree' in subgroup:
                    G = rnd_tree(n)
                else:
                    G = rnd_connected(n)
            except Exception:
                continue

        H = G.copy()
        for _ in range(random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]):
            try:
                H = random.choice(muts)(H)
            except Exception:
                pass
        try:
            H = repair(H, subgroup)
        except Exception:
            continue

        if not (3 <= H.number_of_nodes() <= 50) or not check_graph_class(H, subgroup):
            continue

        try:
            inv = compute_fast(H, needed)
            violation = conjecture.violation(inv)
            score = score_fn(H, inv, conjecture)
            if violation > best_violation:
                best_violation, best_graph, best_inv = violation, H, inv
                if violation > 1e-9:
                    return H, inv, violation, time.time() - start
            pool.append((score, H, inv, violation))
        except Exception:
            continue

        if len(pool) > 40:
            pool.sort(key=lambda x: x[0], reverse=True)
            pool = pool[:20]

    return best_graph, best_inv, best_violation, time.time() - start


# ──────────────────────────────────────────────────────────────
# Évaluation d'un candidat (vecteur de poids) sur un échantillon
# ──────────────────────────────────────────────────────────────

def evaluate_score_function(score_code, conjectures_sample, time_per_conj=8):
    """
    Évalue une fonction heuristic_score (code Python string) sur un échantillon.
    Retourne (total_cost, success_rate, error_msg).
    Interface publique – compatible avec d'éventuels appels externes.
    """
    try:
        ns = {}
        exec(score_code, ns)
        score_fn = ns['heuristic_score']
    except Exception as e:
        return float('inf'), 0.0, str(e)

    total_cost, found = 0.0, 0
    for conj in conjectures_sample:
        try:
            _, _, violation, elapsed = search_with_score_fn(conj, score_fn, time_limit=time_per_conj)
            if violation > 1e-9:
                found += 1
                total_cost += elapsed
            else:
                total_cost += time_per_conj * 2
        except Exception:
            total_cost += time_per_conj * 2

    rate = found / len(conjectures_sample) if conjectures_sample else 0.0
    return total_cost, rate, None


def _evaluate_weights(weights, conjectures_sample, time_per_conj=8):
    """Évalue un vecteur de poids (usage interne)."""
    return evaluate_score_function(weights_to_code(weights), conjectures_sample, time_per_conj)


# ──────────────────────────────────────────────────────────────
# Boucle principale FunSearch (évolutionnaire, sans LLM)
# ──────────────────────────────────────────────────────────────

def funsearch(conjectures, n_iterations=10, sample_size=10,
              time_per_eval=6, population_size=8, verbose=True):
    """
    FunSearch évolutionnaire :
      - Maintient une population de fonctions candidates.
      - Chaque itération : mutation + croisement des meilleurs → nouveaux candidats.
      - Évaluation sur un sous-ensemble → élitisme.
      - Aucun LLM, aucune API.

    Retourne : (best_code, best_cost, history)
    """
    sample = random.sample(conjectures, min(sample_size, len(conjectures)))

    # ── Initialisation de la population ──────────────────────
    population = []   # liste de (cost, rate, weights)

    if verbose:
        print("[FunSearch] Évaluation de la fonction de base...")
    base_cost, base_rate, _ = _evaluate_weights(BASE_WEIGHTS, sample, time_per_eval)
    population.append((base_cost, base_rate, list(BASE_WEIGHTS)))
    if verbose:
        print(f"  Base    : cost={base_cost:.1f}  success={base_rate*100:.0f}%")

    for idx in range(population_size - 1):
        w = random_weights()
        cost, rate, _ = _evaluate_weights(w, sample, time_per_eval)
        population.append((cost, rate, w))
        if verbose:
            print(f"  Seed {idx+1:2d} : cost={cost:.1f}  success={rate*100:.0f}%")

    population.sort(key=lambda x: x[0])
    best_cost, best_rate, best_weights = population[0]

    history = [{
        'iteration': 0,
        'cost': best_cost,
        'success_rate': best_rate,
        'code': weights_to_code(best_weights),
    }]

    if verbose:
        print(f"\n[FunSearch] Init terminée — meilleur: cost={best_cost:.1f}  success={best_rate*100:.0f}%")

    # ── Boucle évolutionnaire ─────────────────────────────────
    for iteration in range(1, n_iterations + 1):
        if verbose:
            print(f"\n[FunSearch] Itération {iteration}/{n_iterations}")

        new_candidates = []

        # Mutation des 3 meilleurs
        for _, _, w in population[:3]:
            new_candidates.append(mutate_weights(w, sigma=0.3))
            new_candidates.append(mutate_weights(w, sigma=0.05))  # mutation fine

        # Croisements
        if len(population) >= 2:
            for _ in range(2):
                p1 = random.choice(population[:4])[2]
                p2 = random.choice(population[:4])[2]
                child = crossover_weights(p1, p2)
                new_candidates.append(mutate_weights(child, sigma=0.1))

        # Injection aléatoire pour diversité
        new_candidates.append(random_weights())

        # Évaluation et mise à jour de la population
        for w in new_candidates:
            cost, rate, _ = _evaluate_weights(w, sample, time_per_eval)
            improved = cost < best_cost
            population.append((cost, rate, w))
            if improved:
                best_cost, best_rate, best_weights = cost, rate, w
            if verbose:
                marker = ' ← NOUVEAU MEILLEUR' if improved else ''
                print(f"  cost={cost:.1f}  success={rate*100:.0f}%{marker}")

        # Élitisme
        population.sort(key=lambda x: x[0])
        population = population[:population_size]

        history.append({
            'iteration': iteration,
            'cost': best_cost,
            'success_rate': best_rate,
            'code': weights_to_code(best_weights),
        })
        if verbose:
            print(f"  → Meilleur courant : cost={best_cost:.1f}  success={best_rate*100:.0f}%")

    return weights_to_code(best_weights), best_cost, history


# ──────────────────────────────────────────────────────────────
# Point d'entrée
# ──────────────────────────────────────────────────────────────

def run_funsearch_benchmark(benchmark_path='benchmark/benchmark.xlsx',
                             n_iterations=10, time_per_eval=6, verbose=True):
    """Lance FunSearch sur l'intégralité du benchmark et sauvegarde les résultats."""
    conjectures = load_benchmark(benchmark_path)
    print(f"=== FunSearch (évolutionnaire, sans API) ===")
    print(f"  {len(conjectures)} conjectures chargées")
    print(f"  {n_iterations} itérations, {time_per_eval}s par évaluation\n")

    best_code, best_cost, history = funsearch(
        conjectures,
        n_iterations=n_iterations,
        sample_size=15,
        time_per_eval=time_per_eval,
        population_size=8,
        verbose=verbose,
    )

    print(f"\n=== Meilleure fonction de score découverte ===")
    print(best_code)

    os.makedirs('results', exist_ok=True)
    with open('results/funsearch_history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)
    with open('results/best_heuristic.py', 'w') as f:
        f.write(best_code + '\n')

    print(f"\nHistorique  → results/funsearch_history.json")
    print(f"Meilleure fn → results/best_heuristic.py")
    return best_code, best_cost, history


if __name__ == '__main__':
    n_iter  = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    t_eval  = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    best_code, best_cost, history = run_funsearch_benchmark(n_iterations=n_iter, time_per_eval=t_eval, verbose=True)
    print(f"\nFunSearch terminé — meilleur coût sur l'échantillon : {best_cost:.1f}")
