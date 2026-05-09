"""
Conjecture parser and graph invariant computation for GraphBench.
"""
import ast
import networkx as nx
import numpy as np
from fractions import Fraction
from itertools import combinations
import pandas as pd


# ──────────────────────────────────────────────────────────────
# Conjecture class
# ──────────────────────────────────────────────────────────────
class Conjecture:
    def __init__(self, row):
        self.id = int(row['Conjecture ID'])
        self.text = row['Conjecture']
        self.subgroup = ast.literal_eval(row['Subgroup'])
        self.x_name = row['X']
        self.y_name = row['Y']
        self.sign = row['Sign']   # '<=' or '>='
        self.degree = int(row['Degree'])
        self.intercept = float(Fraction(str(row['Intercept'])))
        
        coefs_raw = ast.literal_eval(row['Coefficients'])
        self.coefficients = [float(Fraction(c)) for c in coefs_raw]
        # poly_coeffs[i] is the coefficient of x^i
        self.poly_coeffs = [self.intercept] + self.coefficients
    
    def evaluate_bound(self, x_val):
        """Evaluate f(x) = intercept + c1*x + c2*x^2 + ..."""
        return sum(c * (x_val ** i) for i, c in enumerate(self.poly_coeffs))
    
    def violation(self, invariants):
        x_val = invariants.get(self.x_name, 0)
        y_val = invariants.get(self.y_name, 0)
        bound = self.evaluate_bound(x_val)
        if self.sign == '<=':
            return y_val - bound
        else:
            return bound - y_val
    
    def is_violated(self, invariants):
        return self.violation(invariants) > 1e-9
    
    def required_invariant_names(self):
        return {self.x_name, self.y_name}
    
    def __repr__(self):
        return f"Conjecture({self.id}, {self.x_name} vs {self.y_name}, sign={self.sign}, classes={self.subgroup})"


def load_benchmark(path='benchmark/benchmark.xlsx'):
    df = pd.read_excel(path)
    return [Conjecture(row) for _, row in df.iterrows()]


# ──────────────────────────────────────────────────────────────
# Graph invariants
# ──────────────────────────────────────────────────────────────

def _order(G):
    return G.number_of_nodes()

def _size(G):
    return G.number_of_edges()

def _diameter(G):
    if not nx.is_connected(G) or G.number_of_nodes() < 2:
        return float('inf')
    return nx.diameter(G)

def _radius(G):
    if not nx.is_connected(G) or G.number_of_nodes() < 2:
        return float('inf')
    return nx.radius(G)

def _density(G):
    return nx.density(G)

def _minimum_degree(G):
    if G.number_of_nodes() == 0:
        return 0
    return min(d for _, d in G.degree())

def _maximum_degree(G):
    if G.number_of_nodes() == 0:
        return 0
    return max(d for _, d in G.degree())

def _average_degree(G):
    if G.number_of_nodes() == 0:
        return 0
    return sum(d for _, d in G.degree()) / G.number_of_nodes()

def _clique_number(G):
    n = G.number_of_nodes()
    if n <= 60:
        return max(len(c) for c in nx.find_cliques(G))
    best = 0
    for _ in range(10):
        clique = nx.approximation.max_clique(G)
        best = max(best, len(clique))
    return best

def _triangle_number(G):
    triangles = nx.triangles(G)
    return sum(triangles.values()) // 3

def _greedy_domination(G):
    nodes = set(G.nodes())
    dominated = set()
    dom_set = []
    for v in sorted(nodes, key=lambda v: G.degree(v), reverse=True):
        if v not in dominated:
            dom_set.append(v)
            dominated.add(v)
            dominated.update(G.neighbors(v))
        if dominated == nodes:
            break
    return len(dom_set)

def _domination_number(G):
    return _greedy_domination(G)

def _total_domination_number(G):
    if G.number_of_nodes() < 2:
        return 0
    nodes = set(G.nodes())
    dominated = set()
    dom_set = []
    for v in sorted(nodes, key=lambda v: G.degree(v), reverse=True):
        if v not in dominated:
            dom_set.append(v)
            dominated.update(G.neighbors(v))
        if dominated == nodes:
            break
    return len(dom_set)

def _independence_number(G):
    n = G.number_of_nodes()
    if n <= 50:
        complement = nx.complement(G)
        return max(len(c) for c in nx.find_cliques(complement))
    return len(nx.maximal_independent_set(G))

def _vertex_cover_number(G):
    return G.number_of_nodes() - _independence_number(G)

def _independent_domination_number(G):
    best = G.number_of_nodes()
    for _ in range(30):
        mis = nx.maximal_independent_set(G)
        best = min(best, len(mis))
    return best

def _matching_number(G):
    matching = nx.max_weight_matching(G, maxcardinality=True)
    return len(matching)

def _randic_index(G):
    total = 0.0
    for u, v in G.edges():
        du, dv = G.degree(u), G.degree(v)
        if du > 0 and dv > 0:
            total += 1.0 / np.sqrt(du * dv)
    return total

def _harmonic_index(G):
    total = 0.0
    for u, v in G.edges():
        s = G.degree(u) + G.degree(v)
        if s > 0:
            total += 2.0 / s
    return total

def _first_zagreb_index(G):
    return sum(d**2 for _, d in G.degree())

def _second_zagreb_index(G):
    return sum(G.degree(u) * G.degree(v) for u, v in G.edges())

def _proximity(G):
    if not nx.is_connected(G) or G.number_of_nodes() < 2:
        return 0
    n = G.number_of_nodes()
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    prox = float('inf')
    for v in G.nodes():
        s = sum(1.0 / lengths[v][u] for u in G.nodes() if u != v)
        prox = min(prox, s / (n - 1))
    return prox

def _remoteness(G):
    if not nx.is_connected(G) or G.number_of_nodes() < 2:
        return float('inf')
    n = G.number_of_nodes()
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    remote = 0
    for v in G.nodes():
        s = (n-1)/sum(lengths[v][u] for u in G.nodes() if u!=v)
        remote = max(remote, s)
    return remote

def _largest_eigenvalue(G):
    if G.number_of_nodes() < 2:
        return 0
    try:
        A = nx.adjacency_matrix(G).todense().astype(float)
        return float(max(np.linalg.eigvalsh(A)))
    except:
        return 0

def _largest_distance_eigenvalue(G):
    if not nx.is_connected(G) or G.number_of_nodes() < 2:
        return float('inf')
    try:
        D = nx.floyd_warshall_numpy(G)
        return float(max(np.linalg.eigvalsh(D)))
    except:
        return float('inf')

def _second_smallest_laplace_eigenvalue(G):
    if G.number_of_nodes() < 2:
        return 0
    try:
        return float(nx.algebraic_connectivity(G))
    except:
        return 0


INVARIANT_FUNCS = {
    'order': _order, 'size': _size, 'diameter': _diameter,
    'radius': _radius, 'density': _density,
    'minimum_degree': _minimum_degree, 'maximum_degree': _maximum_degree,
    'average_degree': _average_degree, 'clique_number': _clique_number,
    'triangle_number': _triangle_number, 'domination_number': _domination_number,
    'total_domination_number': _total_domination_number,
    'independence_number': _independence_number,
    'vertex_cover_number': _vertex_cover_number,
    'independent_domination_number': _independent_domination_number,
    'matching_number': _matching_number, 'randic_index': _randic_index,
    'harmonic_index': _harmonic_index,
    'first_zagreb_index': _first_zagreb_index,
    'second_zagreb_index': _second_zagreb_index,
    'proximity': _proximity, 'remoteness': _remoteness,
    'largest_eigenvalue': _largest_eigenvalue,
    'largest_distance_eigenvalue': _largest_distance_eigenvalue,
    'second_smallest_laplace_eigenvalue': _second_smallest_laplace_eigenvalue,
}


def compute_invariants(G, needed=None):
    if needed is None:
        needed = INVARIANT_FUNCS.keys()
    result = {}
    for name in needed:
        if name in INVARIANT_FUNCS:
            try:
                result[name] = INVARIANT_FUNCS[name](G)
            except:
                result[name] = 0
    return result


# ──────────────────────────────────────────────────────────────
# Graph class checkers
# ──────────────────────────────────────────────────────────────

def is_connected(G):
    return G.number_of_nodes() > 0 and nx.is_connected(G)

def is_tree(G):
    return nx.is_tree(G)

def is_claw_free(G):
    # Fast path: sort by degree descending — high-degree nodes are most likely claw centres
    for v in sorted(G.nodes(), key=lambda u: G.degree(u), reverse=True):
        neighbors = list(G.neighbors(v))
        d = len(neighbors)
        if d < 3:
            break  # remaining vertices have degree < 3, no claw possible
        # For each pair of non-adjacent neighbors, look for a third non-adjacent one
        non_adj = [(neighbors[i], neighbors[j])
                   for i in range(d) for j in range(i+1, d)
                   if not G.has_edge(neighbors[i], neighbors[j])]
        if len(non_adj) < 2:
            continue
        # Check: any non-adjacent pair that has a third independent neighbor
        nbr_set = set(neighbors)
        for (u1, u2) in non_adj:
            u1_nbrs = set(G.neighbors(u1)) & nbr_set
            u2_nbrs = set(G.neighbors(u2)) & nbr_set
            # A third neighbor u3 that is non-adjacent to both u1 and u2
            candidates = nbr_set - u1_nbrs - u2_nbrs - {u1, u2}
            if candidates:
                return False
    return True

def is_bipartite(G):
    return nx.is_bipartite(G)

def is_planar(G):
    return nx.check_planarity(G)[0]


CLASS_CHECKERS = {
    'connected': is_connected,
    'tree': is_tree,
    'claw_free': is_claw_free,
    'bipartite': is_bipartite,
    'planar': is_planar,
}


def check_graph_class(G, subgroup):
    for cls_name in subgroup:
        if cls_name in CLASS_CHECKERS and not CLASS_CHECKERS[cls_name](G):
            return False
    return True


def to_graph6(G):
    return nx.to_graph6_bytes(G, header=False).decode('ascii').strip()

def from_graph6(s):
    return nx.from_graph6_bytes(s.encode('ascii'))
