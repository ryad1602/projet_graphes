import sys; sys.path.insert(0,'.')
import networkx as nx, numpy as np, random
from solver import ilp_total_domination, exact_independence, is_claw_free

def claw_scores(G, label=''):
    n = G.number_of_nodes()
    if not nx.is_connected(G) or not is_claw_free(G): return
    td = ilp_total_domination(G)
    if td is None: return
    alpha = exact_independence(G)
    mu = len(nx.max_weight_matching(G, maxcardinality=True))
    s6574 = alpha - (td+1)  # sign='<=', y=alpha, bound=1+td; violation when alpha > td+1
    s6582 = (td-1) - mu     # sign='>=', y=mu, bound=td-1; violation when mu < td-1
    s6903 = td - (mu+1)     # sign='<=', y=td, bound=1+mu; violation when td > mu+1
    if max(s6574, s6582, s6903) > -0.5:
        g6 = nx.to_graph6_bytes(G, header=False).decode().strip()
        print('%s: n=%d, td=%d, alpha=%d, mu=%d: 6574=%.2f 6582=%.2f 6903=%.2f g6=%s' % (label,n,td,alpha,mu,s6574,s6582,s6903,g6))

# Square of cycle C_n^2
print('=== Square of cycle C_n^2 ===')
for n in range(5, 25):
    G = nx.power(nx.cycle_graph(n), 2)
    claw_scores(G, 'C%d^2' % n)

# K_k + long path
print()
print('=== K_k + path ===')
for k in [3,4,5,6]:
    for plen in range(5, 25):
        G = nx.complete_graph(k)
        prev = k-1
        for i in range(plen):
            nw = G.number_of_nodes(); G.add_node(nw); G.add_edge(prev, nw); prev = nw
        claw_scores(G, 'K%d+P%d' % (k,plen))

# Circular interval graphs (circulant graphs)
print()
print('=== Circulant graphs C(n, {1,2,...,k}) ===')
for n in range(8, 22):
    for k in range(2, n//2+1):
        offsets = list(range(1, k+1))
        G = nx.circulant_graph(n, offsets)
        claw_scores(G, 'C(%d,%s)' % (n,offsets))

# Two cliques sharing an edge (or vertex), extended
print()
print('=== Two cliques sharing edge ===')
for k in range(3, 10):
    G = nx.Graph()
    for i in range(k):
        for j in range(i+1, k): G.add_edge(i, j)
    for i in range(k-1, 2*k-1):
        for j in range(i+1, 2*k-1): G.add_edge(i, j)
    claw_scores(G, '2*K%d-shared-edge' % k)

# Random claw_free via line graphs of random graphs
print()
print('=== Random line graphs (larger) ===')
for trial in range(2000):
    n_host = random.randint(8, 18)
    p = random.uniform(0.2, 0.6)
    base = nx.erdos_renyi_graph(n_host, p)
    if not nx.is_connected(base): continue
    L = nx.convert_node_labels_to_integers(nx.line_graph(base))
    if L.number_of_nodes() < 10: continue
    claw_scores(L, 'L(G%d)' % trial)

# Random perturbations of K_k
print()
print('=== K_k + multiple paths ===')
for k in [4,5,6]:
    for n_extra in range(5, 20):
        G = nx.complete_graph(k)
        for _ in range(n_extra):
            v = random.randint(0, k-1)
            nw = G.number_of_nodes(); G.add_node(nw); G.add_edge(v, nw)
        claw_scores(G, 'K%d+%dleaves' % (k,n_extra))

print('done')
