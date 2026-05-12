"""
FunSearch - Evolution automatique des poids de mutations.

Tourne indefiniment sur TOUTES les conjectures jusqu'a Ctrl+C.
Sauvegarde best_mutations.py a chaque amelioration.

Usage :
  py src/funsearch.py           # 5s par conjecture, LLM si dispo
  py src/funsearch.py 8         # 8s par conjecture
  py src/funsearch.py 5 unsolved  # seulement les difficiles
"""
import random
import time
import json
import os
import sys

try:
    _env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith('#') and '=' in _line:
                    _k, _v = _line.split('=', 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
except Exception:
    pass

from conjecture import load_benchmark, check_graph_class
from solver import (
    compute_fast, initial_pop, ALL_MUTS, CF_MUTS, TREE_MUTS,
    repair, rnd_claw_free, rnd_connected, rnd_tree,
)


# ──────────────────────────────────────────────────────────────
# Mutations disponibles
# ──────────────────────────────────────────────────────────────

MUT_NAMES = {
    'general':   ['m_add_edge', 'm_rm_edge', 'm_add_node', 'm_rm_node',
                  'm_leaf', 'm_subdivide', 'm_path', 'm_clique',
                  'm_densify', 'm_contract', 'm_twins', 'm_complement_small'],
    'tree':      ['m_leaf', 'm_rm_node', 'm_subdivide', 'm_path'],
    'claw_free': ['m_add_edge', 'm_rm_edge', 'm_add_node',
                  'm_densify', 'm_contract', 'm_twins'],
}
MUT_SIZES = {k: len(v) for k, v in MUT_NAMES.items()}
BASE_MUT_WEIGHTS = {k: [1.0] * MUT_SIZES[k] for k in MUT_NAMES}


# ──────────────────────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────────────────────

def _groq_key(): return os.environ.get('GROQ_API_KEY', '').strip()

def _llm_available():
    if _groq_key():
        try: from groq import Groq; return True  # noqa
        except ImportError: pass
    return False

def _llm_mode_str():
    if _groq_key():
        try: from groq import Groq; return "LLM (GROQ — llama-3.3-70b)"  # noqa
        except ImportError: pass
    return "evolutionnaire (sans API GROQ)"


_LLM_SYSTEM = """\
Tu es un expert en metaheuristiques et theorie des graphes.
Tu optimises les probabilites de selection des mutations d'un recuit simule (SA)
qui cherche des contre-exemples a des conjectures mathematiques sur les graphes.

=== EFFET DE CHAQUE MUTATION SUR LA STRUCTURE DU GRAPHE ===

GENERAL (12) :
  1. m_add_edge        : ajoute une arete aleatoire -> augmente densite, reduit independance
  2. m_rm_edge         : supprime une arete -> reduit densite, peut augmenter independance
  3. m_add_node        : ajoute un sommet avec connexions -> augmente l'ordre
  4. m_rm_node         : supprime un sommet -> reduit l'ordre (cherche contre-exemple minimal)
  5. m_leaf            : ajoute une feuille (degre 1) -> augmente diametre, utile arbres
  6. m_subdivide       : subdivise une arete -> augmente ordre et diametre, reduit degre max
  7. m_path            : ajoute un chemin -> impact fort sur diametre et rayon
  8. m_clique          : cree une clique locale -> densite max locale, augmente nb chromatique
  9. m_densify         : ajoute plusieurs aretes d'un coup -> densification rapide
 10. m_contract        : contracte une arete (fusionne 2 sommets) -> reduit ordre
 11. m_twins           : ajoute un jumeau (meme voisinage) -> affecte domination et independance
 12. m_complement_small: complementaire du graphe si petit -> exploration radicale

TREE (4, arbres uniquement) :
  1. m_leaf      : ajoute feuille (operateur principal de croissance)
  2. m_rm_node   : elagage (supprime feuille)
  3. m_subdivide : modifie profondeur/diametre
  4. m_path      : etend l'arbre avec un chemin

CLAW_FREE (6, graphes sans-griffe, avec reparation automatique) :
  1. m_add_edge  : ajoute arete en maintenant sans-griffe
  2. m_rm_edge   : supprime arete
  3. m_add_node  : ajoute sommet
  4. m_densify   : densification
  5. m_contract  : contraction
  6. m_twins     : jumeau

=== FORMAT DE REPONSE OBLIGATOIRE ===
ANALYSE: <2-3 phrases expliquant quelle strategie adopter et pourquoi>
JSON: {"general":[v1..v12],"tree":[v1..v4],"claw_free":[v1..v6]}

Valeurs entre 0.05 et 5.0. Poids eleve = mutation plus souvent choisie.
"""

_LLM_PROMPT_TPL = """\
=== ETAT ACTUEL (iteration={iteration}) ===
Meilleur cout : {cost:.1f}s | Taux de succes : {rate:.0%}
Historique des couts : {history}

=== POIDS ACTUELS ===
general   [{gnames}] :
  {general}
tree [{tnames}] :
  {tree}
claw_free [{cfnames}] :
  {claw_free}

=== PERFORMANCE PAR CLASSE ===
{class_stats}

Analyse les resultats et propose de nouveaux poids pour ameliorer le score.
Reponds avec ANALYSE puis JSON.
"""


def call_llm(prompt, system=_LLM_SYSTEM):
    from groq import Groq
    r = Groq(api_key=_groq_key()).chat.completions.create(
        model='llama-3.3-70b-versatile',
        messages=[{'role': 'system', 'content': system},
                  {'role': 'user', 'content': prompt}],
        max_tokens=600,
    )
    return r.choices[0].message.content


def generate_with_llm(best_weights, best_cost, best_rate, iteration, cost_history,
                      class_stats=None):
    import re
    try:
        stats_str = class_stats or "  (pas de stats disponibles)"
        prompt = _LLM_PROMPT_TPL.format(
            cost=best_cost, rate=best_rate, iteration=iteration,
            gnames=', '.join(MUT_NAMES['general']),
            tnames=', '.join(MUT_NAMES['tree']),
            cfnames=', '.join(MUT_NAMES['claw_free']),
            general=[round(x, 3) for x in best_weights['general']],
            tree=[round(x, 3) for x in best_weights['tree']],
            claw_free=[round(x, 3) for x in best_weights['claw_free']],
            history=[round(c, 1) for c in cost_history[-6:]],
            class_stats=stats_str,
        )
        raw = call_llm(prompt)

        reasoning = ""
        m_analyse = re.search(r'ANALYSE\s*:\s*(.+?)(?=JSON\s*:|$)', raw, re.DOTALL | re.IGNORECASE)
        if m_analyse:
            reasoning = m_analyse.group(1).strip()

        m_json = re.search(r'JSON\s*:\s*(\{[\s\S]*?\})', raw, re.IGNORECASE)
        if not m_json:
            m_json = re.search(r'\{[\s\S]*\}', raw)
        if not m_json:
            return None, reasoning, "JSON non trouve"
        raw_json = m_json.group(1) if 'JSON' in raw.upper() else m_json.group(0)
        data = json.loads(raw_json)
        for k in MUT_NAMES:
            if k not in data or len(data[k]) != MUT_SIZES[k]:
                return None, reasoning, f"Structure incorrecte pour {k}"
        weights = {k: [max(0.05, min(5.0, float(x))) for x in data[k]] for k in MUT_NAMES}
        return weights, reasoning, None
    except Exception as e:
        return None, "", str(e)


# ──────────────────────────────────────────────────────────────
# Operateurs evolutionnaires
# ──────────────────────────────────────────────────────────────

def random_mut_weights():
    return {k: [abs(random.gauss(1.0, 0.6)) + 0.1 for _ in range(MUT_SIZES[k])]
            for k in MUT_NAMES}

def mutate_mut_weights(weights, sigma=0.3):
    result = {}
    for k in MUT_NAMES:
        w = list(weights[k])
        for i in range(len(w)):
            if random.random() < 0.7:
                w[i] = max(0.05, w[i] + random.gauss(0, sigma * max(w[i], 0.1)))
            if random.random() < 0.1:
                w[i] = abs(random.gauss(1.0, 0.4)) + 0.05
        result[k] = w
    return result

def crossover_mut_weights(w1, w2):
    return {k: [max(0.05, w1[k][i] if random.random() < 0.5 else w2[k][i])
                for i in range(MUT_SIZES[k])]
            for k in MUT_NAMES}

def mut_weights_to_code(weights):
    lines = ['MUTATION_WEIGHTS = {']
    for k in MUT_NAMES:
        vals = ', '.join(f'{x:.4f}' for x in weights[k])
        lines.append(f"    # {', '.join(MUT_NAMES[k])}")
        lines.append(f"    '{k}': [{vals}],")
    lines.append('}')
    return '\n'.join(lines)


# ──────────────────────────────────────────────────────────────
# Recherche avec poids de mutations
# ──────────────────────────────────────────────────────────────

_EXTRA_NEEDED = {
    'order', 'size', 'minimum_degree', 'maximum_degree', 'diameter',
    'density', 'domination_number', 'total_domination_number',
    'independence_number', 'vertex_cover_number', 'matching_number',
    'triangle_number', 'randic_index',
}


def search_with_mut_weights(conjecture, mutation_weights, time_limit=5):
    start = time.time()
    subgroup = conjecture.subgroup
    needed = conjecture.required_invariant_names() | _EXTRA_NEEDED

    if 'tree' in subgroup:
        muts = TREE_MUTS; mut_key = 'tree'
    elif 'claw_free' in subgroup:
        muts = CF_MUTS; mut_key = 'claw_free'
    else:
        muts = ALL_MUTS; mut_key = 'general'

    mut_w = mutation_weights.get(mut_key)

    population = initial_pop(conjecture, size=20)
    best_graph, best_violation, best_inv = None, float('-inf'), None
    pool = []

    for G in population:
        if time.time() - start > time_limit:
            break
        if not check_graph_class(G, subgroup):
            continue
        try:
            inv = compute_fast(G, needed)
            violation = conjecture.violation(inv)
            pool.append((violation, G, inv, violation))
            if violation > best_violation:
                best_violation, best_graph, best_inv = violation, G, inv
                if violation > 1e-9:
                    return G, inv, violation, time.time() - start
        except Exception:
            pass

    pool.sort(key=lambda x: x[0], reverse=True)
    pool = pool[:20]

    while time.time() - start < time_limit:
        if pool:
            _, G, _, _ = pool[random.randint(0, min(4, len(pool) - 1))]
        else:
            n = random.randint(5, 30)
            try:
                G = (rnd_claw_free(n) if 'claw_free' in subgroup
                     else rnd_tree(n) if 'tree' in subgroup
                     else rnd_connected(n))
            except Exception:
                continue

        H = G.copy()
        for _ in range(random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]):
            try:
                H = (random.choices(muts, weights=mut_w)[0] if mut_w
                     else random.choice(muts))(H)
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
            if violation > best_violation:
                best_violation, best_graph, best_inv = violation, H, inv
                if violation > 1e-9:
                    return H, inv, violation, time.time() - start
            pool.append((violation, H, inv, violation))
        except Exception:
            continue

        if len(pool) > 40:
            pool.sort(key=lambda x: x[0], reverse=True)
            pool = pool[:20]

    return best_graph, best_inv, best_violation, time.time() - start


def evaluate_mutation_weights(weights, conjectures, time_per_conj):
    """Evalue sur TOUTES les conjectures. Retourne (cost, rate, class_stats_str)."""
    total_cost, found = 0.0, 0
    by_class = {'general': [0, 0], 'tree': [0, 0], 'claw_free': [0, 0]}
    for conj in conjectures:
        sg = conj.subgroup
        key = 'tree' if 'tree' in sg else ('claw_free' if 'claw_free' in sg else 'general')
        by_class[key][1] += 1
        try:
            _, _, violation, elapsed = search_with_mut_weights(conj, weights, time_per_conj)
            if violation > 1e-9:
                found += 1
                total_cost += elapsed
                by_class[key][0] += 1
            else:
                total_cost += time_per_conj * 2
        except Exception:
            total_cost += time_per_conj * 2
    rate = found / len(conjectures) if conjectures else 0.0
    stats_lines = [f"  {k:10s}: {v[0]}/{v[1]} trouvees" for k, v in by_class.items() if v[1] > 0]
    return total_cost, rate, "\n".join(stats_lines)


# ──────────────────────────────────────────────────────────────
# Sauvegarde / chargement
# ──────────────────────────────────────────────────────────────

def save_best(weights, cost, rate, iteration, history):
    os.makedirs('results', exist_ok=True)
    with open('results/best_mutations.py', 'w') as f:
        f.write(f"# Iteration {iteration} | cost={cost:.1f}s | taux={rate:.0%}\n")
        f.write(mut_weights_to_code(weights) + '\n')
    with open('results/funsearch_history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)


def _load_existing_best():
    """Charge best_mutations.py existant. Retourne (weights, cost) ou (None, inf)."""
    path = 'results/best_mutations.py'
    if not os.path.exists(path):
        return None, float('inf')
    try:
        import re
        with open(path) as f:
            content = f.read()
        m = re.search(r'cost=([\d.]+)s', content)
        existing_cost = float(m.group(1)) if m else float('inf')
        ns = {}
        exec(content, ns)
        weights = ns.get('MUTATION_WEIGHTS')
        if not weights or not all(k in weights for k in MUT_NAMES):
            return None, float('inf')
        return weights, existing_cost
    except Exception:
        return None, float('inf')


# ──────────────────────────────────────────────────────────────
# Boucle principale (tourne jusqu'a Ctrl+C)
# ──────────────────────────────────────────────────────────────

def funsearch(conjectures, time_per_eval=5, population_size=8):
    use_llm = _llm_available()
    n = len(conjectures)
    print(f"[FunSearch] Mode       : {_llm_mode_str()}")
    print(f"[FunSearch] Conjectures: {n} (toutes)")
    print(f"[FunSearch] Temps/eval : {time_per_eval}s x {n} = ~{time_per_eval*n}s par candidat")
    print(f"[FunSearch] Population : {population_size}")
    print(f"[FunSearch] Appuyez Ctrl+C pour arreter proprement\n")

    # ── Chargement du meilleur existant (jamais ecrase si pas mieux) ──
    existing_weights, existing_cost = _load_existing_best()
    if existing_weights:
        print(f"[FunSearch] Best existant charge : cost={existing_cost:.1f}s "
              f"(sera conserve si non ameliore)")
    else:
        print(f"[FunSearch] Aucun best existant — initialisation from scratch")

    # ── Init : evaluer les poids de base + quelques aleatoires ──
    population = []
    cost_history = []

    print("[FunSearch] Initialisation...")
    base_cost, base_rate, base_stats = evaluate_mutation_weights(
        BASE_MUT_WEIGHTS, conjectures, time_per_eval)
    population.append((base_cost, base_rate, {k: list(v) for k, v in BASE_MUT_WEIGHTS.items()}))
    print(f"  Base uniforme : cost={base_cost:.1f}s  taux={base_rate:.0%}")

    if existing_weights:
        ex_cost, ex_rate, ex_stats = evaluate_mutation_weights(existing_weights, conjectures, time_per_eval)
        population.append((ex_cost, ex_rate, existing_weights))
        print(f"  Best existant : cost={ex_cost:.1f}s  taux={ex_rate:.0%}")

    for i in range(population_size - 1):
        w = random_mut_weights()
        cost, rate, _ = evaluate_mutation_weights(w, conjectures, time_per_eval)
        population.append((cost, rate, w))
        print(f"  Seed {i+1:2d}       : cost={cost:.1f}s  taux={rate:.0%}")

    population.sort(key=lambda x: x[0])
    best_cost, best_rate, best_weights = population[0]
    best_class_stats = base_stats
    cost_history.append(best_cost)

    history = [{'iteration': 0, 'cost': best_cost, 'success_rate': best_rate,
                'source': 'init', 'reasoning': '',
                'weights': {k: list(v) for k, v in best_weights.items()}}]

    # Ne sauvegarder que si on fait mieux que le fichier existant
    if best_cost < existing_cost:
        save_best(best_weights, best_cost, best_rate, 0, history)
        print(f"\n[FunSearch] Init OK -- meilleur: cost={best_cost:.1f}s  taux={best_rate:.0%}")
        print(f"[FunSearch] Amelioration -> sauvegarde results/best_mutations.py\n")
    else:
        best_cost = existing_cost
        best_weights = existing_weights
        print(f"\n[FunSearch] Init OK -- best existant conserve : cost={existing_cost:.1f}s\n")

    print(f"{'='*65}\n")

    # ── Boucle infinie ────────────────────────────────────────
    iteration = 0
    try:
        while True:
            iteration += 1
            t_iter = time.time()
            print(f"[Iter {iteration}] Generation des candidats...")

            candidates = []     # list of (label, weights, reasoning)
            llm_reasoning = ""

            if use_llm:
                for attempt in range(2):
                    w, reasoning, err = generate_with_llm(
                        best_weights, best_cost, best_rate, iteration,
                        cost_history, best_class_stats)
                    if w:
                        candidates.append((f'llm_{attempt+1}', w, reasoning))
                        short = reasoning[:120].replace('\n', ' ') if reasoning else ""
                        print(f"  LLM variante {attempt+1} OK")
                        if reasoning:
                            print(f"  -> {short}{'...' if len(reasoning)>120 else ''}")
                        llm_reasoning = reasoning
                    else:
                        print(f"  LLM variante {attempt+1} echec: {err}")
                if not any(src.startswith('llm') for src, _, _ in candidates):
                    use_llm = False
                    print("  -> LLM indisponible, bascule evolutionnaire")

            # Evolutionnaire : toujours en complement (diversite)
            for _, _, w in population[:3]:
                candidates.append(('mut_large', mutate_mut_weights(w, sigma=0.5), ''))
                candidates.append(('mut_fine',  mutate_mut_weights(w, sigma=0.08), ''))
            if len(population) >= 2:
                for _ in range(2):
                    p1 = random.choice(population[:4])[2]
                    p2 = random.choice(population[:4])[2]
                    candidates.append(('cross', mutate_mut_weights(
                        crossover_mut_weights(p1, p2), sigma=0.15), ''))
            candidates.append(('random', random_mut_weights(), ''))

            # ── Evaluation des candidats ───────────────────────
            improved_this_iter = False
            for src, w, reasoning in candidates:
                t0 = time.time()
                cost, rate, cstats = evaluate_mutation_weights(w, conjectures, time_per_eval)
                dt = time.time() - t0
                improved = cost < best_cost
                population.append((cost, rate, w))
                if improved:
                    best_cost, best_rate, best_weights = cost, rate, w
                    best_class_stats = cstats
                    cost_history.append(best_cost)
                    improved_this_iter = True
                    history.append({'iteration': iteration, 'cost': best_cost,
                                    'success_rate': best_rate, 'source': src,
                                    'reasoning': reasoning,
                                    'weights': {k: list(v) for k, v in best_weights.items()}})
                    save_best(best_weights, best_cost, best_rate, iteration, history)
                    print(f"  [{src:12s}] cost={cost:.1f}s  taux={rate:.0%}  "
                          f"eval={dt:.0f}s  *** NOUVEAU MEILLEUR -> sauvegarde ***")
                else:
                    print(f"  [{src:12s}] cost={cost:.1f}s  taux={rate:.0%}  eval={dt:.0f}s")

            population.sort(key=lambda x: x[0])
            population = population[:population_size]

            if not improved_this_iter:
                cost_history.append(best_cost)

            print(f"\n  Meilleur : cost={best_cost:.1f}s  taux={best_rate:.0%}  "
                  f"(iter {iteration}, duree={time.time()-t_iter:.0f}s)\n"
                  f"{'='*65}\n")

    except KeyboardInterrupt:
        print(f"\n[FunSearch] Arrete a l'iteration {iteration}.")
        print(f"[FunSearch] Meilleur cost={best_cost:.1f}s  taux={best_rate:.0%}")
        print(f"[FunSearch] Resultats sauvegardes dans results/best_mutations.py")

    return best_weights, best_cost, history


# ──────────────────────────────────────────────────────────────
# Chargement des conjectures difficiles
# ──────────────────────────────────────────────────────────────

def _load_hard_ids(results_path='results/results.json'):
    """
    Identifie les conjectures ou les poids de mutations ont un impact reel.

    Critere 1 — SA a tourne significativement : time > 6.5s
      (En dessous, le temps est mange par la Phase 1 ILP, pas le SA.)

    Critere 2 — Borderline meme si rapide : time > 1.5s ET violation < 0.05
      (Contre-exemple marginal => le SA a cherche longtemps dans un espace
       proche de la frontiere, tres sensible aux mutations choisies.)

    Critere 3 — Echecs.
    """
    try:
        with open(results_path) as f:
            data = json.load(f)
        hard = []
        for r in data.get('results', []):
            t = r.get('time', 0)
            v = abs(r.get('violation', 0))
            status = r.get('status', '')
            if status != 'OK':
                hard.append(r['conjecture_id'])
            elif t > 6.5:
                hard.append(r['conjecture_id'])
            elif t > 1.5 and v < 0.05:
                hard.append(r['conjecture_id'])
        return hard
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────
# Point d'entree
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    t_eval = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    benchmark_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 'benchmark', 'benchmark.xlsx')
    all_conjectures = load_benchmark(benchmark_path)

    if len(sys.argv) > 2 and sys.argv[2] == 'unsolved':
        hard_ids = set(_load_hard_ids())
        if hard_ids:
            conjectures = [c for c in all_conjectures if c.id in hard_ids]
            others = [c for c in all_conjectures if c.id not in hard_ids]
            conjectures += random.sample(others, min(20, len(others)))
            print(f"Mode cible : {len(hard_ids)} difficiles + 20 aleatoires = {len(conjectures)}")
        else:
            conjectures = all_conjectures
    else:
        conjectures = all_conjectures

    funsearch(conjectures, time_per_eval=t_eval, population_size=8)
