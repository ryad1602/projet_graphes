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
        +5.0036 * violation
        +0.3698 * diam
        +0.4322 * Delta
        +0.4076 * delta
        -0.0415 * n
        +0.0336 * m
        -0.3290 * density
        -0.2705 * triangles
        -0.4431 * alpha
        -0.4011 * tau
        -0.3648 * mu
        +0.3692 * td
        -0.1108 * gamma
        -0.3656 * randic
        +0.4121 * td_minus_mu
        +0.0371 * tau_minus_td
        +0.3003 * alpha_ratio
        +0.1505 * deg_spread
    )
