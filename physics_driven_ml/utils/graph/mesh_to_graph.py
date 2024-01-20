from adjacency_dofs import build_dof_adjacency


def build_adjacency(mesh, V):
    dmplex = mesh.topology_dm
    section = V.dm.getLocalSection()
    # Get number of dofs
    ndofs = V.dof_count
    # Get number of entities of each dimension
    dimension = mesh.geometric_dimension()
    # Total number of mesh entities
    num_entities = sum(mesh.num_entities(d) for d in range(dimension + 1))
    return build_dof_adjacency(ndofs, num_entities, dmplex, section)
