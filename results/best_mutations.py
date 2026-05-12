# Iteration 0 | cost=21.8s | taux=100%
MUTATION_WEIGHTS = {
    # m_add_edge, m_rm_edge, m_add_node, m_rm_node, m_leaf, m_subdivide, m_path, m_clique, m_densify, m_contract, m_twins, m_complement_small
    'general': [0.7379, 1.8320, 1.8178, 0.1886, 0.7792, 1.2911, 0.6974, 1.6005, 1.4375, 0.9807, 1.7633, 1.2098],
    # m_leaf, m_rm_node, m_subdivide, m_path
    'tree': [0.5021, 1.0787, 1.4343, 0.8381],
    # m_add_edge, m_rm_edge, m_add_node, m_densify, m_contract, m_twins
    'claw_free': [1.3758, 0.9933, 1.0557, 0.7701, 1.1578, 1.2238],
}
