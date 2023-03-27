import numpy as np

from upyog._compat import iteritems

def create_sparse_stoichiometric_matrix(model, dtype = np.float64):
    metabolites   = model.metabolites
    reactions     = model.reactions

    n_metabolites = len(metabolites)
    n_reactions   = len(reactions)

    matrix        = np.zeros((n_metabolites, n_reactions * 2), dtype = dtype) # account for reversible reactions too.

    m_index       = metabolites.index
    r_index       = reactions.index

    offset        = 0

    for i, reaction in enumerate(model.reactions):
        reaction_index = i * 2

        for metabolite, coeff in iteritems(reaction.metabolites):
            matrix[m_index(metabolite), reaction_index]     =  coeff
            matrix[m_index(metabolite), reaction_index + 1] = -coeff

    return matrix