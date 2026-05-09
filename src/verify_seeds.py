import sys; sys.path.insert(0,'.')
import networkx as nx, numpy as np
from solver import ilp_total_domination, exact_independence
import pulp

def score_all(G, label=''):
    n = G.number_of_nodes()
    td = ilp_total_domination(G)
    if td is None: return
    alpha = exact_independence(G)
    vc = n - alpha
    mu = len(nx.max_weight_matching(G, maxcardinality=True))
    ri = sum(1.0/np.sqrt(G.degree(u)*G.degree(v)) for u,v in G.edges() if G.degree(u)>0 and G.degree(v)>0)
    s1566 = (td-1) - alpha
    s1708 = td - (alpha+1)
    s1587 = (2/3*td - 1/3) - mu
    s1600 = (2/3*td - 1/3) - vc
    s1891 = td - (1.5*vc + 0.5)
    s2120 = td - (1.5*mu + 0.5)
    ic,c1,c2,c3 = 6929257/963766, 1308631/631598, -72533/159523, 18001/599095
    bound2252 = ic + c1*ri + c2*ri**2 + c3*ri**3
    s2252 = td - bound2252
    print('%s: n=%d, td=%d, vc=%d, mu=%d, randic=%.3f' % (label,n,td,vc,mu,ri))
    print('  1566=%.3f 1708=%.3f 1587=%.3f 1600=%.3f 1891=%.3f 2120=%.3f 2252=%.3f' % (s1566,s1708,s1587,s1600,s1891,s2120,s2252))
    viols = [k for k,v in [('1566',s1566),('1708',s1708),('1587',s1587),('1600',s1600),('1891',s1891),('2120',s2120),('2252',s2252)] if v>0]
    if viols: print('  VIOLATIONS: %s' % viols)

def bridge_star(k, pend=1):
    G = nx.Graph()
    node = k+1
    for i in range(1, k+1):
        G.add_edge(0, node); G.add_edge(i, node); node += 1
    for h in range(k+1):
        for _ in range(pend):
            G.add_edge(h, node); node += 1
    return G

for k in [2,3,4,5,6]:
    for pend in [1,2]:
        G = bridge_star(k, pend)
        score_all(G, 'bridge_star(k=%d,pend=%d)' % (k,pend))

print()
G1708 = nx.from_graph6_bytes(b'OdOGEC??G@_N?N??_?GCG')
score_all(G1708, '1708_counterex')

print()
print('=== 2051 asymmetric double stars ===')
for k1,k2 in [(5,6),(5,7),(5,8),(6,7),(7,8)]:
    G = nx.Graph()
    G.add_edge(0,1)
    for i in range(2, k1+2): G.add_edge(0,i)
    for i in range(k1+2, k1+k2+2): G.add_edge(1,i)
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    prob = pulp.LpProblem('idom', pulp.LpMinimize)
    x = {v: pulp.LpVariable('x%d'%v, cat='Binary') for v in nodes}
    prob += pulp.lpSum(x[v] for v in nodes)
    for v in nodes:
        prob += pulp.lpSum(x[u] for u in [v]+list(G.neighbors(v))) >= 1
    for u,v in G.edges(): prob += x[u]+x[v] <= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    idom = int(pulp.value(prob.objective))
    alpha = exact_independence(G)
    vc = n - alpha
    s2051 = (3/4 + 1/4*idom) - vc
    print('  S_%d,%d: n=%d, idom=%d, vc=%d: 2051=%.3f' % (k1,k2,n,idom,vc,s2051))
