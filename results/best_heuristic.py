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
        +12.0758 * violation
        -0.0412 * diam
        +0.0171 * Delta
        +0.4536 * delta
        -0.2442 * m
        +0.0737 * density
        +0.0539 * alpha
        -0.3117 * tau
        -0.1030 * mu
        +0.1787 * td
        -0.0392 * randic
        +0.4123 * td_minus_mu
        +0.0649 * tau_minus_td
        -0.0952 * alpha_ratio
        +0.2694 * deg_spread
    )
