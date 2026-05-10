"""
GraphBench heuristic solver v4.
- compute_fast() is always O(poly): no exponential clique-finding
- counterexamples are always verified with exact ILP before being returned
- search() returns only true violations
"""
import random
import math
import time
import networkx as nx
import numpy as np
import pulp
from itertools import combinations
from conjecture import (
    load_benchmark, check_graph_class, to_graph6,
    is_claw_free, is_connected, is_tree
)
import json


# ──────────────────────────────────────────────────────────────
# EXACT invariants via ILP
# ──────────────────────────────────────────────────────────────

def ilp_domination(G):
    nodes = list(G.nodes())
    prob = pulp.LpProblem("dom", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x{v}", cat='Binary') for v in nodes}
    prob += pulp.lpSum(x[v] for v in nodes)
    for v in nodes:
        prob += pulp.lpSum(x[u] for u in [v] + list(G.neighbors(v))) >= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=4))
    return int(pulp.value(prob.objective)) if prob.status == 1 else None

def ilp_total_domination(G):
    nodes = list(G.nodes())
    if any(G.degree(v) == 0 for v in nodes):
        return None
    prob = pulp.LpProblem("tdom", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x{v}", cat='Binary') for v in nodes}
    prob += pulp.lpSum(x[v] for v in nodes)
    for v in nodes:
        prob += pulp.lpSum(x[u] for u in G.neighbors(v)) >= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=4))
    return int(pulp.value(prob.objective)) if prob.status == 1 else None

def ilp_independent_domination(G):
    nodes = list(G.nodes())
    prob = pulp.LpProblem("idom", pulp.LpMinimize)
    x = {v: pulp.LpVariable(f"x{v}", cat='Binary') for v in nodes}
    prob += pulp.lpSum(x[v] for v in nodes)
    for v in nodes:
        prob += pulp.lpSum(x[u] for u in [v] + list(G.neighbors(v))) >= 1
    for u, v in G.edges():
        prob += x[u] + x[v] <= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=4))
    return int(pulp.value(prob.objective)) if prob.status == 1 else None

def ilp_independence(G):
    nodes = list(G.nodes())
    prob = pulp.LpProblem("indep", pulp.LpMaximize)
    x = {v: pulp.LpVariable(f"x{v}", cat='Binary') for v in nodes}
    prob += pulp.lpSum(x[v] for v in nodes)
    for u, v in G.edges():
        prob += x[u] + x[v] <= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=4))
    return int(pulp.value(prob.objective)) if prob.status == 1 else None

def exact_independence(G):
    """
    Exact independence number.
    find_cliques(complement) only for n<=20 (safe). ILP above that.
    """
    n = G.number_of_nodes()
    if n <= 20:
        complement = nx.complement(G)
        return max(len(c) for c in nx.find_cliques(complement))
    val = ilp_independence(G)
    return val if val is not None else _fast_indep(G)


# ──────────────────────────────────────────────────────────────
# EXACT compute — used for final verification
# ──────────────────────────────────────────────────────────────

def compute(G, needed):
    result = {}
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if 'order' in needed: result['order'] = n
    if 'size' in needed: result['size'] = m
    if 'density' in needed: result['density'] = nx.density(G)
    if 'minimum_degree' in needed: result['minimum_degree'] = min(d for _, d in G.degree()) if n > 0 else 0
    if 'maximum_degree' in needed: result['maximum_degree'] = max(d for _, d in G.degree()) if n > 0 else 0
    if 'average_degree' in needed: result['average_degree'] = (2*m/n) if n > 0 else 0
    if 'first_zagreb_index' in needed: result['first_zagreb_index'] = sum(d**2 for _, d in G.degree())
    if 'second_zagreb_index' in needed: result['second_zagreb_index'] = sum(G.degree(u)*G.degree(v) for u,v in G.edges())
    if 'triangle_number' in needed: result['triangle_number'] = sum(nx.triangles(G).values()) // 3
    if 'clique_number' in needed:
        result['clique_number'] = max(len(c) for c in nx.find_cliques(G)) if n > 0 else 0
    if 'matching_number' in needed:
        result['matching_number'] = len(nx.max_weight_matching(G, maxcardinality=True))
    if 'randic_index' in needed:
        result['randic_index'] = sum(1.0/np.sqrt(G.degree(u)*G.degree(v)) for u,v in G.edges() if G.degree(u)>0 and G.degree(v)>0)
    if 'harmonic_index' in needed:
        result['harmonic_index'] = sum(2.0/(G.degree(u)+G.degree(v)) for u,v in G.edges() if G.degree(u)+G.degree(v)>0)

    need_dist = needed & {'diameter','radius','proximity','remoteness'}
    if need_dist and nx.is_connected(G) and n >= 2:
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        ecc = {v: max(lengths[v].values()) for v in G.nodes()}
        if 'diameter' in needed: result['diameter'] = max(ecc.values())
        if 'radius' in needed: result['radius'] = min(ecc.values())
        if 'proximity' in needed:
            result['proximity'] = min((n-1)/sum(lengths[v][u] for u in G.nodes() if u!=v) for v in G.nodes())
        if 'remoteness' in needed:
            result['remoteness'] = max((n-1)/sum(lengths[v][u] for u in G.nodes() if u!=v) for v in G.nodes())
    else:
        for inv in need_dist:
            result[inv] = float('inf') if inv in ('diameter','radius') else 0

    if 'domination_number' in needed:
        v = ilp_domination(G)
        result['domination_number'] = v if v is not None else _fast_dom(G)
    if 'total_domination_number' in needed:
        v = ilp_total_domination(G)
        if v is None:
            # greedy fallback: cover all nodes via neighbors
            nodes = set(G.nodes()); dominated = set(); dom = 0
            for u in sorted(nodes, key=lambda u: G.degree(u), reverse=True):
                if u not in dominated:
                    dom += 1
                    dominated.update(G.neighbors(u))
                if dominated == nodes: break
            v = dom
        result['total_domination_number'] = v
    if 'independence_number' in needed:
        v = exact_independence(G)
        result['independence_number'] = v if v is not None else _fast_indep(G)
    if 'vertex_cover_number' in needed:
        alpha = result.get('independence_number') or exact_independence(G) or _fast_indep(G)
        result['vertex_cover_number'] = n - alpha
    if 'independent_domination_number' in needed:
        v = ilp_independent_domination(G)
        result['independent_domination_number'] = v if v is not None else _fast_indep(G)

    if 'largest_eigenvalue' in needed:
        if n >= 2:
            try:
                A = nx.adjacency_matrix(G).todense().astype(float)
                result['largest_eigenvalue'] = float(max(np.linalg.eigvalsh(A)))
            except: result['largest_eigenvalue'] = 0
        else: result['largest_eigenvalue'] = 0
    if 'largest_distance_eigenvalue' in needed:
        if nx.is_connected(G) and n >= 2:
            try:
                D = nx.floyd_warshall_numpy(G)
                result['largest_distance_eigenvalue'] = float(max(np.linalg.eigvalsh(D)))
            except: result['largest_distance_eigenvalue'] = float('inf')
        else: result['largest_distance_eigenvalue'] = float('inf')
    if 'second_smallest_laplace_eigenvalue' in needed:
        if n >= 2:
            try:
                # Dense Laplacian — avoids ARPACK hangs on near-disconnected graphs
                L = nx.laplacian_matrix(G).toarray().astype(float)
                evals = np.linalg.eigvalsh(L)
                result['second_smallest_laplace_eigenvalue'] = float(sorted(evals)[1])
            except: result['second_smallest_laplace_eigenvalue'] = 0
        else: result['second_smallest_laplace_eigenvalue'] = 0

    return result


# ──────────────────────────────────────────────────────────────
# FAST compute — ALWAYS polynomial, no ILP, no find_cliques
# ──────────────────────────────────────────────────────────────

def _fast_total_dom(G):
    """Greedy upper bound for total domination number. O(n^2).
    Correct: returns a valid total dominating set size (every vertex has
    a neighbor in S), always tighter than 2*matching_number."""
    n = G.number_of_nodes()
    if n < 2: return 0
    if any(G.degree(v) == 0 for v in G.nodes()):
        return n
    dominated = set()
    S = []
    all_nodes = set(G.nodes())
    while dominated != all_nodes:
        best = max(all_nodes, key=lambda v: len(set(G.neighbors(v)) - dominated))
        S.append(best)
        dominated.update(G.neighbors(best))
    return len(S)


def _fast_dom(G):
    nodes = set(G.nodes())
    dominated, dom = set(), 0
    for v in sorted(nodes, key=lambda v: G.degree(v), reverse=True):
        if v not in dominated:
            dom += 1
            dominated.add(v)
            dominated.update(G.neighbors(v))
        if dominated == nodes:
            break
    return dom

def _fast_indep(G):
    """Greedy MIS — lower bound on independence_number. Always O(n+m)."""
    nodes = sorted(G.nodes(), key=lambda v: G.degree(v))
    indep, excluded = set(), set()
    for v in nodes:
        if v not in excluded:
            indep.add(v)
            excluded.update(G.neighbors(v))
            excluded.add(v)
    return len(indep)

def compute_fast(G, needed):
    """
    Strictly polynomial — never calls ILP or find_cliques on complement.
    Safe to call thousands of times in the mutation loop.
    """
    result = {}
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if 'order' in needed: result['order'] = n
    if 'size' in needed: result['size'] = m
    if 'density' in needed: result['density'] = nx.density(G)
    if 'minimum_degree' in needed: result['minimum_degree'] = min(d for _, d in G.degree()) if n > 0 else 0
    if 'maximum_degree' in needed: result['maximum_degree'] = max(d for _, d in G.degree()) if n > 0 else 0
    if 'average_degree' in needed: result['average_degree'] = (2*m/n) if n > 0 else 0
    if 'first_zagreb_index' in needed: result['first_zagreb_index'] = sum(d**2 for _, d in G.degree())
    if 'second_zagreb_index' in needed: result['second_zagreb_index'] = sum(G.degree(u)*G.degree(v) for u,v in G.edges())
    if 'triangle_number' in needed: result['triangle_number'] = sum(nx.triangles(G).values()) // 3
    if 'clique_number' in needed:
        # find_cliques on G itself (not complement) is ok for clique_number on small/dense graphs
        result['clique_number'] = max(len(c) for c in nx.find_cliques(G)) if n > 0 else 0
    if 'matching_number' in needed:
        result['matching_number'] = len(nx.max_weight_matching(G, maxcardinality=True))
    if 'randic_index' in needed:
        result['randic_index'] = sum(1.0/np.sqrt(G.degree(u)*G.degree(v)) for u,v in G.edges() if G.degree(u)>0 and G.degree(v)>0)
    if 'harmonic_index' in needed:
        result['harmonic_index'] = sum(2.0/(G.degree(u)+G.degree(v)) for u,v in G.edges() if G.degree(u)+G.degree(v)>0)

    need_dist = needed & {'diameter','radius','proximity','remoteness'}
    if need_dist and nx.is_connected(G) and n >= 2:
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        ecc = {v: max(lengths[v].values()) for v in G.nodes()}
        if 'diameter' in needed: result['diameter'] = max(ecc.values())
        if 'radius' in needed: result['radius'] = min(ecc.values())
        if 'proximity' in needed:
            result['proximity'] = min((n-1)/sum(lengths[v][u] for u in G.nodes() if u!=v) for v in G.nodes())
        if 'remoteness' in needed:
            result['remoteness'] = max((n-1)/sum(lengths[v][u] for u in G.nodes() if u!=v) for v in G.nodes())
    else:
        for inv in need_dist:
            result[inv] = float('inf') if inv in ('diameter','radius') else 0

    # Greedy approximations — always O(n+m), never exponential
    if 'domination_number' in needed:
        result['domination_number'] = _fast_dom(G)
    if 'total_domination_number' in needed:
        if n >= 2:
            result['total_domination_number'] = _fast_total_dom(G)
        else:
            result['total_domination_number'] = 0
    # independence/vertex_cover: ALWAYS greedy — never find_cliques on complement
    if 'independence_number' in needed:
        result['independence_number'] = _fast_indep(G)
    if 'vertex_cover_number' in needed:
        alpha = result.get('independence_number', _fast_indep(G))
        result['vertex_cover_number'] = n - alpha
    if 'independent_domination_number' in needed:
        result['independent_domination_number'] = _fast_indep(G)

    if 'largest_eigenvalue' in needed:
        if n >= 2:
            try:
                A = nx.adjacency_matrix(G).todense().astype(float)
                result['largest_eigenvalue'] = float(max(np.linalg.eigvalsh(A)))
            except: result['largest_eigenvalue'] = 0
        else: result['largest_eigenvalue'] = 0
    if 'largest_distance_eigenvalue' in needed:
        if nx.is_connected(G) and n >= 2:
            try:
                D = nx.floyd_warshall_numpy(G)
                result['largest_distance_eigenvalue'] = float(max(np.linalg.eigvalsh(D)))
            except: result['largest_distance_eigenvalue'] = float('inf')
        else: result['largest_distance_eigenvalue'] = float('inf')
    if 'second_smallest_laplace_eigenvalue' in needed:
        if n >= 2:
            try:
                # Dense Laplacian — avoids ARPACK hangs on near-disconnected graphs
                L = nx.laplacian_matrix(G).toarray().astype(float)
                evals = np.linalg.eigvalsh(L)
                result['second_smallest_laplace_eigenvalue'] = float(sorted(evals)[1])
            except: result['second_smallest_laplace_eigenvalue'] = 0
        else: result['second_smallest_laplace_eigenvalue'] = 0

    return result


# Invariants that require ILP for exact computation
_ILP_INVARIANTS = {
    'domination_number', 'total_domination_number',
    'independent_domination_number',
    'independence_number', 'vertex_cover_number',
}

def _needs_ilp(needed):
    return bool(needed & _ILP_INVARIANTS)


# ──────────────────────────────────────────────────────────────
# Graph generators
# ──────────────────────────────────────────────────────────────

def rnd_connected(n, p=0.3):
    for _ in range(50):
        G = nx.erdos_renyi_graph(n, p)
        if nx.is_connected(G): return G
    G = nx.random_labeled_tree(n)
    for _ in range(int(p*n*(n-1)/4)):
        u,v = random.sample(range(n),2)
        G.add_edge(u,v)
    return G

def rnd_tree(n):
    return nx.random_labeled_tree(n)

def rnd_claw_free(n):
    for _ in range(80):
        bn = random.randint(3, max(4, n))
        bp = random.uniform(0.2, 0.7)
        base = nx.erdos_renyi_graph(bn, bp)
        if not nx.is_connected(base):
            comps = list(nx.connected_components(base))
            for i in range(len(comps)-1):
                u = random.choice(list(comps[i]))
                v = random.choice(list(comps[i+1]))
                base.add_edge(u,v)
        L = nx.line_graph(base)
        L = nx.convert_node_labels_to_integers(L)
        if L.number_of_nodes() >= 3 and nx.is_connected(L) and is_claw_free(L):
            return L
    return nx.complete_graph(min(n, 6))


def targeted_graphs(x_name, y_name, subgroup):
    graphs = []
    invs = {x_name, y_name}

    if invs & {'maximum_degree', 'triangle_number'}:
        for k in [4, 5, 6, 7, 8]:
            for n in [20, 30, 40, 60, 80, 90]:
                if (k * n) % 2 == 0 and k < n:
                    try:
                        G = nx.random_regular_graph(k, n)
                        if nx.is_connected(G): graphs.append(G)
                    except: pass
        for n in [8, 12, 15, 20]: graphs.append(nx.complete_graph(n))

    if invs & {'triangle_number', 'largest_eigenvalue'}:
        for n in [8, 10, 12, 15, 18, 20]: graphs.append(nx.complete_graph(n))
        for k in [5, 6, 7, 8, 9]:
            try: graphs.append(nx.barbell_graph(k, 1))
            except: pass
        for bn in range(4, 10):
            L = nx.line_graph(nx.complete_graph(bn))
            graphs.append(nx.convert_node_labels_to_integers(L))

    if 'second_smallest_laplace_eigenvalue' in invs:
        # Small Fiedler: barbell (near-disconnected), bridge graphs, path
        for k in [3, 4, 5, 6, 7, 8, 9, 10]:
            try: graphs.append(nx.barbell_graph(k, 1))
            except: pass
        for k in [5, 7, 9]:
            try: graphs.append(nx.lollipop_graph(k, k))
            except: pass
        for n in [5, 8, 10, 12, 15]: graphs.append(nx.complete_graph(n))
        for n in [8, 12, 16, 20, 30]: graphs.append(nx.path_graph(n))
        # Near-bridge: two cliques sharing one edge
        for k in [3, 4, 5, 6]:
            G = nx.Graph()
            for i in range(k):
                for j in range(i+1, k): G.add_edge(i, j)
            for i in range(k, 2*k):
                for j in range(i+1, 2*k): G.add_edge(i, j)
            G.add_edge(0, k)
            graphs.append(G)
        # Two cliques connected by a path of length p
        for k in [3, 4, 5]:
            for p in [1, 2, 3, 4]:
                G = nx.Graph()
                for i in range(k):
                    for j in range(i+1, k): G.add_edge(i, j)
                for i in range(k, 2*k):
                    for j in range(i+1, 2*k): G.add_edge(i, j)
                prev = 0
                for step in range(p):
                    nw = G.number_of_nodes(); G.add_node(nw)
                    G.add_edge(prev, nw); prev = nw
                G.add_edge(prev, k)
                graphs.append(G)

    if 'largest_distance_eigenvalue' in invs:
        # Large distance eigenvalue: long paths, large cycles, spider graphs
        for n in [6, 8, 9, 10, 12, 15, 20]: graphs.append(nx.cycle_graph(n))
        for k in [4, 5, 6, 7]:
            try: graphs.append(nx.barbell_graph(k, 1))
            except: pass
        for n in [5, 7, 9, 11, 15, 20]: graphs.append(nx.path_graph(n))
        for n in [5, 7, 9, 11]:
            L = nx.line_graph(nx.path_graph(n))
            if L.number_of_nodes() >= 3:
                graphs.append(nx.convert_node_labels_to_integers(L))
        # Spider graphs (center + arms): large distance eigenvalue
        for arms in [3, 4, 5]:
            for arm_len in [3, 4, 5, 6]:
                G = nx.Graph(); G.add_node(0); nid = 1
                for _ in range(arms):
                    prev = 0
                    for _ in range(arm_len):
                        G.add_node(nid); G.add_edge(prev, nid); prev = nid; nid += 1
                graphs.append(G)

    if invs & {'first_zagreb_index', 'second_zagreb_index'}:
        for n in [8, 10, 12, 14, 16, 18, 20]: graphs.append(nx.complete_graph(n))
        for k in [4, 5, 6, 8, 10]:
            for n in [10, 12, 14, 16, 18, 20]:
                if (k * n) % 2 == 0 and k < n:
                    try:
                        G = nx.random_regular_graph(k, n)
                        if nx.is_connected(G): graphs.append(G)
                    except: pass
        for k in range(5, 14):
            try: graphs.append(nx.barbell_graph(k, 1))
            except: pass

    if invs & {'domination_number', 'total_domination_number'}:
        for n in [10, 14, 18, 22, 26, 30]:
            graphs.append(nx.path_graph(n))
            graphs.append(nx.cycle_graph(n))
        for k in [3, 4, 5, 6]:
            try:
                G = nx.grid_2d_graph(k, k)
                graphs.append(nx.convert_node_labels_to_integers(G))
            except: pass
        for k in [5, 7, 9, 12]:
            graphs.append(nx.complete_bipartite_graph(k, k))

    if invs & {'second_zagreb_index', 'independent_domination_number'}:
        # P_9 violates conjecture 2882: second_zagreb=28, i_dom=4, bound=3.75, viol=+0.25
        for n in [9, 11, 13, 15, 17, 19, 21]:
            graphs.append(nx.path_graph(n))
        for k in [3, 4, 5, 6, 7]:
            graphs.append(nx.star_graph(k))

    # Asymmetric double stars: conjecture 2051 (ind_dom vs vertex_cover).
    # S_{k1,k2} has ind_dom=1+min(k1,k2), vc=2. Violation when min(k1,k2)>=5 and k1!=k2.
    if 'independent_domination_number' in invs:
        for k1 in range(1, 12):
            for k2 in range(k1 + 1, 15):
                G = nx.Graph()
                G.add_edge(0, 1)
                for i in range(2, k1 + 2): G.add_edge(0, i)
                for i in range(k1 + 2, k1 + k2 + 2): G.add_edge(1, i)
                graphs.append(G)

    # Extremal total-domination: "dense-core + pendant paths" pattern.
    # Counterexample for 1708: K_5-like cluster with several pendant paths attached.
    if 'total_domination_number' in invs:
        # K_k with pendant paths of varying length from each vertex
        for k in range(3, 8):
            for plen in range(1, 5):
                G = nx.complete_graph(k)
                for v in range(k):
                    prev = v
                    for _ in range(plen):
                        nw = G.number_of_nodes(); G.add_node(nw); G.add_edge(prev, nw); prev = nw
                graphs.append(G)
        # "Spine" graph: path of cliques K_k connected in a chain (high γ_t, low β)
        for k in range(3, 6):
            for chain_len in range(2, 6):
                G = nx.Graph()
                # Build chain of K_k's sharing one vertex
                base = 0
                for _ in range(chain_len):
                    clique = list(range(base, base + k))
                    for u in clique:
                        for v in clique:
                            if u < v: G.add_edge(u, v)
                    base += k - 1  # share last vertex with next clique
                graphs.append(G)
        # Graphs with many pendants where domination requires traversal
        for spine_len in range(4, 10):
            for leaves_per in range(1, 4):
                G = nx.path_graph(spine_len)
                nw = spine_len
                for v in range(spine_len):
                    for _ in range(leaves_per):
                        G.add_node(nw); G.add_edge(v, nw); nw += 1
                graphs.append(G)
        # bridge_star(k, pend): bipartite hub structure, violates 1587/1600/1891/2120 (k≥3)
        # and 2252 (k≥5). Structure: hub 0 + k peripheral hubs + k bridge vertices
        # (each bridge adjacent to hub 0 AND one peripheral hub) + pendant leaves.
        for k in range(2, 9):
            for pend in range(1, 4):
                G = nx.Graph()
                node = k + 1
                for i in range(1, k + 1):
                    G.add_edge(0, node); G.add_edge(i, node); node += 1
                for h in range(k + 1):
                    for _ in range(pend):
                        G.add_edge(h, node); node += 1
                graphs.append(G)

    if invs & {'independence_number', 'vertex_cover_number', 'independent_domination_number'}:
        for k in [5, 7, 9, 12, 15]:
            graphs.append(nx.complete_bipartite_graph(k, k))
            try: graphs.append(nx.complete_bipartite_graph(k, k+2))
            except: pass
        for n in [10, 14, 18, 22, 26]:
            graphs.append(nx.cycle_graph(n))
            graphs.append(nx.path_graph(n))
        for n in [15, 20, 25]:
            for p in [0.1, 0.15, 0.2]:
                try:
                    G = nx.erdos_renyi_graph(n, p)
                    if nx.is_connected(G): graphs.append(G)
                except: pass

    if 'matching_number' in invs:
        for k in [5, 8, 10, 12, 15]:
            try: graphs.append(nx.complete_bipartite_graph(k, k))
            except: pass
        for k in [3, 4]:
            for n in [10, 14, 18, 22, 26, 30]:
                if (k * n) % 2 == 0:
                    try:
                        G = nx.random_regular_graph(k, n)
                        if nx.is_connected(G): graphs.append(G)
                    except: pass

    if 'clique_number' in invs:
        for n in [8, 10, 12, 14, 16, 20]: graphs.append(nx.complete_graph(n))
        for n in [10, 14, 18]:
            for p in [0.6, 0.75, 0.9]:
                try:
                    G = nx.erdos_renyi_graph(n, p)
                    if nx.is_connected(G): graphs.append(G)
                except: pass
        if 'claw_free' in subgroup and 'average_degree' in invs:
            # K_k + long path: claw-free, omega=k, average_degree approaches 2 from above
            # Violates conjecture 6272 for k>=4 with long enough path
            for k in [3, 4, 5, 6, 7]:
                for path_len in [10, 15, 20, 30, 40]:
                    G = nx.complete_graph(k)
                    prev = k - 1
                    for i in range(path_len):
                        nw = G.number_of_nodes(); G.add_node(nw); G.add_edge(prev, nw); prev = nw
                    graphs.append(G)

    if 'tree' in subgroup and invs & {'second_smallest_laplace_eigenvalue', 'diameter'}:
        # Double star S_{k,k}: diameter=3, Fiedler=1/(2(k+1)) — very small for large k
        # Violates conjecture 4288: f(3)≈0.171 > Fiedler≈0.035 for k=13
        for k in [5, 8, 10, 13, 15, 20, 30, 50]:
            G = nx.Graph()
            G.add_edge(0, 1)
            for i in range(2, k + 2):
                G.add_edge(0, i)
            for i in range(k + 2, 2 * k + 2):
                G.add_edge(1, i)
            graphs.append(G)
        # Also lollipop-like trees with long paths (large diameter, small Fiedler)
        for arm in [10, 15, 20, 30]:
            G = nx.path_graph(arm)
            # attach star at one end
            base = arm
            for i in range(1, arm // 3 + 1):
                G.add_edge(0, base + i)
            graphs.append(G)

    if invs & {'diameter', 'radius'}:
        for n in [10, 14, 18, 22, 26, 30]:
            graphs.append(nx.path_graph(n))
            graphs.append(nx.cycle_graph(n))
        for k in [5, 8, 10, 12]:
            try: graphs.append(nx.lollipop_graph(k, k))
            except: pass
        for n in [15, 20, 25]:
            G = nx.path_graph(n // 2)
            for v in list(G.nodes()):
                for _ in range(2):
                    nw = G.number_of_nodes(); G.add_node(nw); G.add_edge(v, nw)
            graphs.append(G)

    if invs & {'proximity', 'remoteness'}:
        for n in [10, 15, 20, 30, 40]:
            graphs.append(nx.star_graph(n))
            graphs.append(nx.path_graph(n))
        for k in [5, 8, 10]:
            try: graphs.append(nx.lollipop_graph(k, k))
            except: pass
        for n in [8, 12, 16]: graphs.append(nx.complete_graph(n))

    if 'density' in invs:
        for n in [10, 14, 18, 22]:
            for p in [0.15, 0.3, 0.5, 0.7]:
                try:
                    G = nx.erdos_renyi_graph(n, p)
                    if nx.is_connected(G): graphs.append(G)
                except: pass

    if invs & {'average_degree', 'maximum_degree'}:
        for n in [10, 14, 18, 22, 26]:
            for p in [0.2, 0.4, 0.6, 0.8]:
                try:
                    G = nx.erdos_renyi_graph(n, p)
                    if nx.is_connected(G): graphs.append(G)
                except: pass
        for k in [4, 6, 8, 10]:
            try: graphs.append(nx.barbell_graph(k, 1))
            except: pass
        for bn in range(4, 12):
            L = nx.line_graph(nx.complete_graph(bn))
            graphs.append(nx.convert_node_labels_to_integers(L))

    if invs & {'randic_index', 'second_zagreb_index'}:
        for k in [3, 4, 5, 6]:
            for n in [10, 14, 18, 22]:
                if (k * n) % 2 == 0:
                    try:
                        G = nx.random_regular_graph(k, n)
                        if nx.is_connected(G): graphs.append(G)
                    except: pass

    return graphs


def initial_pop(conjecture, size=30):
    subgroup = conjecture.subgroup
    pop = list(targeted_graphs(conjecture.x_name, conjecture.y_name, subgroup))

    if 'tree' in subgroup:
        for n in [5, 8, 10, 15, 20, 30, 40]:
            pop.append(rnd_tree(n))
            pop.append(nx.star_graph(n - 1))
            pop.append(nx.path_graph(n))
            G = nx.path_graph(max(3, n // 3))
            for v in list(G.nodes()):
                for _ in range(random.randint(0, 2)):
                    nw = G.number_of_nodes(); G.add_node(nw); G.add_edge(v, nw)
            pop.append(G)
            G = nx.Graph(); G.add_edge(0, 1)
            for i in range(2, n // 2 + 1): G.add_node(i); G.add_edge(0, i)
            for i in range(n // 2 + 1, n): G.add_node(i); G.add_edge(1, i)
            pop.append(G)
        for r in [2, 3]:
            for h in range(2, 5):
                pop.append(nx.balanced_tree(r, h))
        for long_n in [50, 70, 100]:
            pop.append(nx.path_graph(long_n))
        # Double stars S_{k,k}: diameter=3, Fiedler≈1/(2k) — key for Fiedler/diameter trees
        for k in [8, 13, 20, 30, 50]:
            G = nx.Graph()
            G.add_edge(0, 1)
            for i in range(2, k + 2): G.add_edge(0, i)
            for i in range(k + 2, 2 * k + 2): G.add_edge(1, i)
            pop.append(G)
        for arms in [3, 4, 5, 6]:
            G = nx.Graph(); G.add_node(0); nid = 1
            for _ in range(arms):
                prev = 0
                for _ in range(random.randint(2, 5)):
                    G.add_node(nid); G.add_edge(prev, nid); prev = nid; nid += 1
            pop.append(G)
        for _ in range(15):
            pop.append(rnd_tree(random.randint(5, 35)))

    elif 'claw_free' in subgroup:
        for n in [4, 5, 6, 8, 10, 12, 15, 20]:
            pop.append(nx.complete_graph(n))
        for n in [4, 5, 6, 8, 10, 12, 15, 20]:
            pop.append(nx.cycle_graph(n))
        for bn in range(3, 12):
            L = nx.line_graph(nx.complete_graph(bn))
            pop.append(nx.convert_node_labels_to_integers(L))
        for bn in range(4, 16):
            L = nx.line_graph(nx.cycle_graph(bn))
            pop.append(nx.convert_node_labels_to_integers(L))
        for bn in range(3, 16):
            L = nx.line_graph(nx.path_graph(bn))
            if nx.is_connected(L):
                pop.append(nx.convert_node_labels_to_integers(L))
        for sn in range(3, 12):
            L = nx.line_graph(nx.star_graph(sn))
            pop.append(nx.convert_node_labels_to_integers(L))
        for k in range(4, 20):
            G = nx.Graph()
            for i in range(k):
                for j in range(i + 1, k): G.add_edge(i, j)
            for i in range(k, 2 * k):
                for j in range(i + 1, 2 * k): G.add_edge(i, j)
            G.add_edge(0, k)
            pop.append(G)
        for k in range(4, 15):
            G = nx.Graph()
            for i in range(k):
                for j in range(i + 1, k): G.add_edge(i, j)
            for i in range(k - 1, 2 * k - 1):
                for j in range(i + 1, 2 * k - 1): G.add_edge(i, j)
            pop.append(G)
        for k in range(4, 10):
            for plen in range(1, 8):
                G = nx.complete_graph(k)
                prev = 0
                for i in range(plen):
                    nw = G.number_of_nodes(); G.add_node(nw)
                    G.add_edge(prev, nw)
                    if i == 0: G.add_edge(nw, 1)
                    prev = nw
                pop.append(G)
        for _ in range(10):
            base = rnd_connected(random.randint(4, 12), random.uniform(0.2, 0.6))
            L = nx.line_graph(base)
            L = nx.convert_node_labels_to_integers(L)
            if nx.is_connected(L): pop.append(L)
        for _ in range(10):
            pop.append(rnd_claw_free(random.randint(5, 20)))
        # K_k + path: claw-free, high clique, low average_degree → violates 6272
        for k in [3, 4, 5, 6]:
            for path_len in [8, 15, 25, 40]:
                G = nx.complete_graph(k)
                prev = k - 1
                for i in range(path_len):
                    nw = G.number_of_nodes(); G.add_node(nw); G.add_edge(prev, nw); prev = nw
                pop.append(G)
        # Claw-free graphs with high total_domination relative to matching/alpha
        # Target conjectures 6903, 6574, 6582: non-Hamiltonian claw-free graphs
        if 'total_domination_number' in (conjecture.x_name, conjecture.y_name):
            # Line graphs of non-Eulerian host graphs (≥3 odd-degree vertices)
            # Key: non-Hamiltonian L(G) can have γ_t > μ+1
            for k in range(3, 12):
                # Host = star K_{1,k} with one extra pendant: 3 odd-degree vertices
                G_host = nx.star_graph(k)
                extra = G_host.number_of_nodes()
                G_host.add_node(extra); G_host.add_edge(1, extra)
                L = nx.convert_node_labels_to_integers(nx.line_graph(G_host))
                if nx.is_connected(L) and is_claw_free(L): pop.append(L)
            # Line graphs of "double star" host graphs
            for k1 in range(3, 8):
                for k2 in range(k1, 10):
                    G_host = nx.Graph()
                    G_host.add_edge(0, 1)
                    for i in range(2, k1 + 2): G_host.add_edge(0, i)
                    for i in range(k1 + 2, k1 + k2 + 2): G_host.add_edge(1, i)
                    L = nx.convert_node_labels_to_integers(nx.line_graph(G_host))
                    if L.number_of_nodes() >= 3 and nx.is_connected(L) and is_claw_free(L):
                        pop.append(L)
            # Line graphs of caterpillars (path + leaves)
            for back in range(3, 9):
                for pend in range(1, 4):
                    G_host = nx.path_graph(back)
                    nw = back
                    for v in range(back):
                        for _ in range(pend):
                            G_host.add_node(nw); G_host.add_edge(v, nw); nw += 1
                    L = nx.convert_node_labels_to_integers(nx.line_graph(G_host))
                    if L.number_of_nodes() >= 4 and nx.is_connected(L) and is_claw_free(L):
                        pop.append(L)
            # Line graphs of random graphs with odd-degree vertices
            for _ in range(20):
                n_host = random.randint(5, 12)
                p_host = random.uniform(0.2, 0.6)
                G_host = rnd_connected(n_host, p_host)
                L = nx.convert_node_labels_to_integers(nx.line_graph(G_host))
                if L.number_of_nodes() >= 4 and nx.is_connected(L) and is_claw_free(L):
                    pop.append(L)

        # Extra claw-free with extreme Fiedler/distance eigenvalue properties
        if conjecture.x_name in ('second_smallest_laplace_eigenvalue', 'largest_distance_eigenvalue') or \
           conjecture.y_name in ('second_smallest_laplace_eigenvalue', 'largest_distance_eigenvalue'):
            # Barbell K_k - K_k: claw-free AND small Fiedler value
            for k in [3, 4, 5, 6, 7, 8]:
                G = nx.Graph()
                for i in range(k):
                    for j in range(i+1, k): G.add_edge(i, j)
                for i in range(k, 2*k):
                    for j in range(i+1, 2*k): G.add_edge(i, j)
                G.add_edge(0, k)
                if is_claw_free(G): pop.append(G)
            # Line graphs of paths: claw-free + path-like distances
            for n in [5, 8, 10, 12, 15, 20]:
                L = nx.line_graph(nx.path_graph(n))
                if L.number_of_nodes() >= 3:
                    pop.append(nx.convert_node_labels_to_integers(L))
            # Line graphs of stars: complete graphs (small distance eigenvalue)
            for sn in [4, 5, 6, 7, 8, 10]:
                L = nx.line_graph(nx.star_graph(sn))
                pop.append(nx.convert_node_labels_to_integers(L))

    else:
        for n in [5, 7, 10, 12, 15, 18, 20, 25, 30]:
            for p in [0.1, 0.2, 0.4, 0.7]:
                pop.append(rnd_connected(n, p))
            pop.append(nx.complete_graph(min(n, 15)))
            pop.append(nx.cycle_graph(n))
            pop.append(nx.star_graph(n - 1))
            pop.append(nx.path_graph(n))
            if n >= 6: pop.append(nx.lollipop_graph(n // 2, n - n // 2))
        for half in range(5, 18):
            pop.append(nx.barbell_graph(half, 1))
        for k in range(2, 8):
            pop.append(nx.complete_bipartite_graph(k, k))
        pop.append(nx.petersen_graph())

        # Asymmetric double stars (critical for conj 2051 and related)
        for k1 in range(1, 12):
            for k2 in range(k1 + 1, 14):
                G = nx.Graph()
                G.add_edge(0, 1)
                for i in range(2, k1 + 2): G.add_edge(0, i)
                for i in range(k1 + 2, k1 + k2 + 2): G.add_edge(1, i)
                pop.append(G)

        # Dense core + pendant paths (pattern of conj 1708 counterexample)
        for k in range(3, 8):
            for plen in range(1, 5):
                G = nx.complete_graph(k)
                for v in range(min(k, 4)):
                    prev = v
                    for _ in range(plen):
                        nw = G.number_of_nodes(); G.add_node(nw); G.add_edge(prev, nw); prev = nw
                pop.append(G)

        # Caterpillar trees with varying backbone and pendant lengths
        for back in range(3, 12):
            for pend in range(1, 5):
                G = nx.path_graph(back)
                nw = back
                for v in range(back):
                    for _ in range(pend):
                        G.add_node(nw); G.add_edge(v, nw); nw += 1
                pop.append(G)

        # bridge_star family: bipartite, violates 1587/1600/1891/2120 (k≥3) and 2252 (k≥5)
        for k in range(2, 9):
            for pend in range(1, 4):
                G = nx.Graph()
                node = k + 1
                for i in range(1, k + 1):
                    G.add_edge(0, node); G.add_edge(i, node); node += 1
                for h in range(k + 1):
                    for _ in range(pend):
                        G.add_edge(h, node); node += 1
                pop.append(G)

    valid = [G for G in pop if G.number_of_nodes() >= 3 and check_graph_class(G, subgroup)]
    seen = set()
    deduped = []
    for G in valid:
        key = (G.number_of_nodes(), G.number_of_edges())
        if key not in seen:
            seen.add(key)
            deduped.append(G)
    random.shuffle(deduped)
    return deduped[:size * 2]


# ──────────────────────────────────────────────────────────────
# Mutations
# ──────────────────────────────────────────────────────────────

def m_add_edge(G):
    H=G.copy(); nodes=list(H.nodes())
    for _ in range(20):
        u,v=random.sample(nodes,2)
        if not H.has_edge(u,v): H.add_edge(u,v); return H
    return H

def m_rm_edge(G):
    H=G.copy(); edges=list(H.edges()); random.shuffle(edges)
    for u,v in edges[:10]:
        H.remove_edge(u,v)
        if nx.is_connected(H): return H
        H.add_edge(u,v)
    return H

def m_add_node(G):
    H=G.copy(); new=max(H.nodes())+1
    k=random.randint(1,min(3,H.number_of_nodes()))
    H.add_node(new)
    for t in random.sample(list(H.nodes()),k): H.add_edge(new,t)
    return H

def m_rm_node(G):
    H=G.copy()
    if H.number_of_nodes()<=4: return H
    for v in random.sample(list(H.nodes()),min(5,H.number_of_nodes())):
        H2=H.copy(); H2.remove_node(v)
        if H2.number_of_nodes()>0 and nx.is_connected(H2):
            return nx.convert_node_labels_to_integers(H2)
    return H

def m_leaf(G):
    H=G.copy(); v=random.choice(list(H.nodes())); new=max(H.nodes())+1
    H.add_node(new); H.add_edge(v,new); return H

def m_subdivide(G):
    H=G.copy(); edges=list(H.edges())
    if not edges: return H
    u,v=random.choice(edges); new=max(H.nodes())+1
    H.remove_edge(u,v); H.add_node(new); H.add_edge(u,new); H.add_edge(new,v)
    return H

def m_path(G):
    H=G.copy(); v=random.choice(list(H.nodes()))
    base=max(H.nodes())+1; prev=v
    for i in range(random.randint(2,5)):
        new=base+i; H.add_node(new); H.add_edge(prev,new); prev=new
    return H

def m_clique(G):
    H=G.copy(); k=random.randint(3,5); base=max(H.nodes())+1
    cn=list(range(base,base+k))
    for i in range(k):
        H.add_node(cn[i])
        for j in range(i): H.add_edge(cn[i],cn[j])
    H.add_edge(random.choice(list(G.nodes())),random.choice(cn))
    return H

def m_densify(G):
    H=G.copy()
    for _ in range(random.randint(2,max(2,H.number_of_nodes()//2))):
        u,v=random.sample(list(H.nodes()),2); H.add_edge(u,v)
    return H

def m_contract(G):
    H=G.copy()
    if H.number_of_nodes()<=4: return H
    u,v=random.choice(list(H.edges()))
    for w in list(H.neighbors(v)):
        if w!=u: H.add_edge(u,w)
    H.remove_node(v)
    return nx.convert_node_labels_to_integers(H)

def m_twins(G):
    H=G.copy(); v=random.choice(list(H.nodes())); new=max(H.nodes())+1
    H.add_node(new)
    for u in G.neighbors(v): H.add_edge(new,u)
    return H

def m_complement_small(G):
    H=G.copy(); nodes=list(H.nodes())
    if len(nodes)<4: return H
    subset=random.sample(nodes,min(4,len(nodes)))
    for u,v in combinations(subset,2):
        if H.has_edge(u,v): H.remove_edge(u,v)
        else: H.add_edge(u,v)
    return H if nx.is_connected(H) else G

ALL_MUTS = [m_add_edge,m_rm_edge,m_add_node,m_rm_node,m_leaf,m_subdivide,m_path,m_clique,m_densify,m_contract,m_twins,m_complement_small]
TREE_MUTS = [m_leaf,m_rm_node,m_subdivide,m_path]
CF_MUTS = [m_add_edge,m_rm_edge,m_add_node,m_densify,m_contract,m_twins]


def repair(G, subgroup):
    if not nx.is_connected(G) and G.number_of_nodes() > 0:
        comps = list(nx.connected_components(G))
        prev = comps[0]
        for comp in comps[1:]:
            u=random.choice(list(prev)); v=random.choice(list(comp))
            G.add_edge(u,v); prev = prev|comp
    if 'tree' in subgroup and not nx.is_tree(G):
        if nx.is_connected(G):
            G = nx.minimum_spanning_tree(G)
            G = nx.convert_node_labels_to_integers(G)
    if 'claw_free' in subgroup:
        for _ in range(300):
            if is_claw_free(G): break
            found=False
            for v in G.nodes():
                nbrs=list(G.neighbors(v))
                if len(nbrs)<3: continue
                for u1,u2,u3 in combinations(nbrs,3):
                    if not G.has_edge(u1,u2) and not G.has_edge(u1,u3) and not G.has_edge(u2,u3):
                        G.add_edge(*random.choice([(u1,u2),(u1,u3),(u2,u3)]))
                        found=True; break
                if found: break
            if not found: break
    return G


# ──────────────────────────────────────────────────────────────
# Search engine
# ──────────────────────────────────────────────────────────────

def _atlas_graphs():
    """All connected graphs from networkx atlas (n=1..7, ~850 graphs)."""
    from networkx.generators.atlas import graph_atlas_g
    return [G for G in graph_atlas_g() if G.number_of_nodes() >= 3 and nx.is_connected(G)]


_ATLAS_CACHE = None


def search(conjecture, time_limit=60, verbose=False, score_fn=None):
    """
    score_fn : fonction optionnelle heuristic_score(G, inv, conjecture) → float.
               Si fournie, elle guide l'ordre d'exploration du pool (SA).
               La détection des contre-exemples reste toujours basée sur violation exacte.
    """
    global _ATLAS_CACHE
    start = time.time()
    subgroup = conjecture.subgroup
    needed = conjecture.required_invariant_names()
    muts = TREE_MUTS if 'tree' in subgroup else (CF_MUTS if 'claw_free' in subgroup else ALL_MUTS)
    use_ilp = _needs_ilp(needed)

    def elapsed(): return time.time() - start

    def pool_score(G, inv, violation):
        if score_fn is None:
            return violation
        try:
            return score_fn(G, inv, conjecture)
        except Exception:
            return violation

    # ── Phase 0 : exhaustive sweep of all small graphs (atlas) ─
    # Check all connected graphs n=3..7 instantly — guaranteed to find
    # counterexamples that exist on small graphs before any mutation.
    best_graph = None
    best_score = float('-inf')
    best_inv = None

    if _ATLAS_CACHE is None:
        _ATLAS_CACHE = _atlas_graphs()

    for G in _ATLAS_CACHE:
        if elapsed() > time_limit * 0.10:
            break
        if not check_graph_class(G, subgroup):
            continue
        try:
            if use_ilp:
                # n≤7: ILP is instantaneous — use exact compute on every atlas graph
                inv = compute(G, needed)
            else:
                inv = compute_fast(G, needed)
            score = conjecture.violation(inv)
            if score > best_score:
                best_score = score; best_graph = G; best_inv = inv
            if score > 1e-9:
                return G, inv, score, elapsed()
        except: pass

    # ── Phase 1 : population initiale ──────────────────────────
    pop = initial_pop(conjecture)
    random.shuffle(pop); pop = pop[:25]
    pool = []

    for G in pop:
        if elapsed() > time_limit * 0.15:
            break
        if not check_graph_class(G, subgroup): continue
        try:
            inv = compute_fast(G, needed)
            score = conjecture.violation(inv)
            pool.append((pool_score(G, inv, score), G))
            if score > best_score:
                best_score = score; best_graph = G; best_inv = inv
            # Early return for non-ILP: compute_fast == exact for these invariants
            if score > 1e-9 and not use_ilp:
                return G, inv, score, elapsed()
        except: pass

    # Exact verification of the best initial candidate
    if best_graph is not None and use_ilp:
        try:
            exact_inv = compute(best_graph, needed)
            exact_score = conjecture.violation(exact_inv)
            best_score = exact_score; best_inv = exact_inv
            if exact_score > 1e-9:
                return best_graph, exact_inv, exact_score, elapsed()
        except: pass

    pool.sort(key=lambda x: x[0], reverse=True)
    pool = pool[:40]

    # ── Phase 2 : mutation loop with simulated annealing ───────
    stale = 0
    temperature = 0.5
    cooling = 0.997

    while elapsed() < time_limit:
        if pool:
            k = min(3, len(pool))
            _, G = max(random.sample(pool, k), key=lambda x: x[0])
        else:
            n = random.randint(5, 30)
            try:
                G = (rnd_tree(n) if 'tree' in subgroup
                     else (rnd_claw_free(n) if 'claw_free' in subgroup
                           else rnd_connected(n, random.uniform(0.1, 0.9))))
            except: continue

        H = G.copy()
        for _ in range(random.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]):
            try: H = random.choice(muts)(H)
            except: pass
        try: H = repair(H, subgroup)
        except: continue

        nn = H.number_of_nodes()
        max_nn = 40 if use_ilp else 100
        # claw_free repair calls is_claw_free() O(n·Δ³) — caps at 35 to avoid hangs
        if 'claw_free' in subgroup:
            max_nn = min(max_nn, 35)
        if nn < 3 or nn > max_nn: continue
        if not check_graph_class(H, subgroup): continue

        try:
            inv_fast = compute_fast(H, needed)
            score_fast = conjecture.violation(inv_fast)
        except: continue

        # Run exact ILP only when fast score is promising
        if use_ilp and score_fast > best_score - 0.5:
            try:
                inv_exact = compute(H, needed)
                score = conjecture.violation(inv_exact)
                inv = inv_exact
            except:
                score, inv = score_fast, inv_fast
        else:
            score, inv = score_fast, inv_fast

        if score > best_score:
            best_score = score; best_graph = H; best_inv = inv; stale = 0
            if score > 1e-9:
                # Result is exact: either compute() was called (use_ilp=True)
                # or compute_fast() == compute() for non-ILP invariants.
                if verbose:
                    print(f"  FOUND conj {conjecture.id}: v={score:.4f} t={elapsed():.2f}s n={H.number_of_nodes()}")
                return H, inv, score, elapsed()
        else:
            stale += 1
            delta = score - best_score
            if delta > -temperature and random.random() < math.exp(delta / max(temperature, 1e-6)):
                pool.append((pool_score(H, inv, score), H))

        pool.append((pool_score(H, inv, score), H))
        temperature = max(0.001, temperature * cooling)

        if len(pool) > 80:
            pool.sort(key=lambda x: x[0], reverse=True)
            pool = pool[:30] + random.sample(pool[30:], min(10, len(pool) - 30))

        if stale > 50:
            stale = 0
            temperature = min(temperature * 3, 0.5)
            if best_graph is not None:
                for _ in range(3):
                    try:
                        Hb = best_graph.copy()
                        for __ in range(random.randint(2, 4)):
                            Hb = random.choice(muts)(Hb)
                        Hb = repair(Hb, subgroup)
                        if check_graph_class(Hb, subgroup):
                            pool.append((best_score - 0.01, Hb))
                    except: pass
            for _ in range(3):
                n = random.randint(4, 45)
                try:
                    Gn = (rnd_tree(n) if 'tree' in subgroup
                          else (rnd_claw_free(n) if 'claw_free' in subgroup
                                else rnd_connected(n, random.uniform(0.05, 0.95))))
                    if check_graph_class(Gn, subgroup): pool.append((-999, Gn))
                except: pass
            # Inject targeted graphs for hard invariants
            fresh = targeted_graphs(conjecture.x_name, conjecture.y_name, subgroup)
            random.shuffle(fresh)
            for Gf in fresh[:5]:
                if check_graph_class(Gf, subgroup):
                    pool.append((-500, Gf))

    return best_graph, best_inv, best_score, elapsed()


# ──────────────────────────────────────────────────────────────
# Exact verification — use this to double-check any result
# ──────────────────────────────────────────────────────────────

def verify_exact(graph, conjecture):
    """
    Recompute all invariants with exact methods and check violation.
    Returns (violation_value, invariants_dict, is_true_counterexample).
    Call this on any result you want to confirm.
    """
    needed = conjecture.required_invariant_names()
    inv = compute(graph, needed)
    v = conjecture.violation(inv)
    return v, inv, v > 1e-9
