def heuristic_score(G, invariants, conjecture):
    """
    G : graphe NetworkX
    invariants : dictionnaire des invariants calcules
    conjecture : objet decrivant la conjecture
    retourne un score numerique a maximiser
    """
    violation = conjecture.violation(invariants)
    n         = invariants.get("order", 0)
    m         = invariants.get("size", 0)
    delta     = invariants.get("minimum_degree", 0)
    Delta     = invariants.get("maximum_degree", 0)
    diam      = invariants.get("diameter", 0)
    gamma     = invariants.get("domination_number", 0)
    alpha     = invariants.get("independence_number", 0)
    tau       = invariants.get("vertex_cover_number", 0)
    triangles = invariants.get("triangle_number", 0)
    mu        = invariants.get("matching_number", 0)
    td        = invariants.get("total_domination_number", 0)
    density   = invariants.get("density", 0)
    randic    = invariants.get("randic_index", 0)
    td_minus_mu  = td - mu
    tau_minus_td = tau - td
    alpha_ratio  = alpha / n if n > 0 else 0
    deg_spread   = Delta - delta
    return (
        +10.0183 * violation
        +0.0460 * diam
        -0.3977 * delta
        +0.2590 * n
        -0.3478 * m
        +0.4939 * density
        +0.1560 * alpha
        +0.2412 * tau
        +0.1946 * mu
        +0.0047 * td
        -0.1534 * gamma
        -0.2721 * randic
        -0.1955 * tau_minus_td
        +0.9027 * alpha_ratio
        -0.0727 * deg_spread
    )
