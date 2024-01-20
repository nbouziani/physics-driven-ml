# Which one is faster: getAdjacency for every entity or use hashmap ?

# cython: language_level=3

from firedrake.petsc import PETSc
from firedrake.utils import IntType

import numpy as np

cimport cython
cimport numpy as np
cimport petsc4py.PETSc as PETSc

include "petschdr.pxi"

@cython.boundscheck(False)
@cython.wraparound(False)
def build_dof_adjacency(ndofs, num_entities, dmplex, section):
    cdef:
        PetscInt i, j, k, dof, size_entity_adj, max_adjacent_dofs
        np.ndarray[PetscInt, ndim=1, mode="c"] number_adj_entities
        np.ndarray[PetscInt, ndim=1, mode="c"] adj_entity_offset
        np.ndarray[PetscInt, ndim=1, mode="c"] entity_per_dof_count
        np.ndarray[PetscInt, ndim=1, mode="c"] dof_to_entity_offset
        np.ndarray[PetscInt, ndim=1, mode="c"] adj_entities
        np.ndarray[PetscInt, ndim=1, mode="c"] dofs_to_entities
        np.ndarray[PetscInt, ndim=1, mode="c"] dofs_idx
        np.ndarray[PetscInt, ndim=2, mode="c"] adjacency_dofs

    # Number of adjacent entities of each entity
    number_adj_entities = np.zeros(num_entities + 1, dtype=IntType)
    # Offset of each entity in `adj_entities`
    adj_entity_offset = np.empty(num_entities + 1, dtype=IntType)
    # Number of entities associated with each dof
    entity_per_dof_count = np.zeros(ndofs + 1, dtype=IntType)
    # Offset of each dof in `dofs_to_entities`
    dof_to_entity_offset = np.empty(ndofs + 1, dtype=IntType)
    # Mapping from dof to entity
    dofs_to_entities = np.empty(ndofs, dtype=IntType)
    # Helper array to populate `dofs_to_entities`
    dofs_idx = np.zeros(ndofs, dtype=IntType)
    # Adjacency list of degrees of freedom
    # -> 100 is the upper bound on the number of adjacent dofs
    adjacency_dofs = np.empty((100 * ndofs, 2), dtype=IntType)


    # Preprocessing loop: populate `number_adj_entities` and `entity_per_dof_count`
    size_entity_adj = 0
    for i in range(num_entities):
        # Number of adjacent entities of entity i
        number_adj_entities[i + 1] = dmplex.getAdjacency(i).size
        size_entity_adj += number_adj_entities[i + 1]
        # Number of dofs associated with entity i
        dof_offset = section.getOffset(i) + 1
        for n in range(section.getDof(i)):
            entity_per_dof_count[dof_offset + n] += 1

    # Compute offsets
    dof_to_entity_offset = np.cumsum(entity_per_dof_count, dtype=IntType)
    adj_entity_offset = np.cumsum(number_adj_entities, dtype=IntType)
    # Free memory of both: `entity_per_dof_count` and `number_adj_entities`
    del entity_per_dof_count, number_adj_entities

    # Adjacency list of entities
    adj_entities = np.empty(size_entity_adj, dtype=IntType)

    # Populate `adj_entities`, `dofs_to_entities`, and `dofs_idx`
    for i in range(num_entities):
        # Build adjacency list of entities
        # adj_entities:
        #                 entity i     
        #  | --------- | --------- | --------- |
        #              ^           ^
        #              |           |
        #              a           b
        # with `a = adj_entity_offset[i]` and  `b = adj_entity_offset[i + 1]`
        start_ent, end_ent = adj_entity_offset[i], adj_entity_offset[i+1]
        adj_i = dmplex.getAdjacency(i)
        for j in range(start_ent, end_ent):
            adj_entities[j] = adj_i[j - start_ent]
        # Build mapping from dofs to entities
        # dofs_to_entities:
        #                 dof j
        # | --------- | --------- | --------- |
        #             ^           ^
        #             |           |
        #             a           b
        # with `a = dof_to_entity_offset[j]` and  `b = dof_to_entity_offset[j + 1]`
        start_dof = section.getOffset(i)
        end_dof = start_dof + section.getDof(i)
        for dof in range(start_dof, end_dof):
            dof_offset = dof_to_entity_offset[dof]
            dofs_to_entities[dof_offset + dofs_idx[dof]] = i
            dofs_idx[dof] += 1

    # Populate `adjacency_dofs`
    max_adjacent_dofs = 0
    for i in range(ndofs):
        # Loop over entities related to `dof`
        for j in range(dof_to_entity_offset[i], dof_to_entity_offset[i + 1]):
            e = dofs_to_entities[j]
            # Loop over adjacent entities of `e`
            start_ent, end_ent = adj_entity_offset[e], adj_entity_offset[e+1]
            for k in range(start_ent, end_ent):
                ek = adj_entities[k]
                # Loop over dofs associated with `ek`
                start_dof = section.getOffset(ek)
                end_dof = start_dof + section.getDof(ek)
                for dof in range(start_dof, end_dof):
                    # Add edge to adjacency list if not already present
                    if i >= dof:
                        adjacency_dofs[max_adjacent_dofs, 0] = dof
                        adjacency_dofs[max_adjacent_dofs, 1] = i
                        max_adjacent_dofs += 1

    return adjacency_dofs[:max_adjacent_dofs, :]
