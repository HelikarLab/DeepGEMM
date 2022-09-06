from bpyutils._compat import iteritems

from cobra.flux_analysis import (
    pfba,
    find_blocked_reactions
)
from cobra.util.solver import (
    linear_reaction_coefficients
)

from gempy.model.util import (
    find_reaction,
    find_mle_reactions
)

from bpyutils.util.array import squash

def minreact(model, growth_rate_cutoff = 1):
    reactions = linear_reaction_coefficients(model)

    objective_value = model.slim_optimize()

    lower_bound = growth_rate_cutoff * objective_value

    with model:
        for reaction, coeff in iteritems(reactions):
            reaction = model.reactions.get_by_id(reaction.id)
            reaction.lower_bound = lower_bound

            pfba_solution = pfba(model)

            remove_reaction_ids = find_blocked_reactions(model)
            zero_flux_rxn_ids = []
            
            for i, flux in enumerate(pfba_solution.fluxes):
                if flux == 0:
                    reaction = model.reactions[i]
                    zero_flux_rxn_ids.append(reaction.id)
                
            remove_reaction_ids += zero_flux_rxn_ids
            remove_reaction_ids += find_mle_reactions(model, zero_flux_rxns = model.reactions.get_by_any(zero_flux_rxn_ids))

            model.remove_reactions(remove_reaction_ids)

        return model

MINIMIZATION_ALGORITHMS = {
    "minreact": {
        "fn": minreact
    }
}

def minimize_model(model, algorithm = "minreact"):
    if algorithm not in MINIMIZATION_ALGORITHMS:
        raise ValueError("Not a valid algorithm, accepted are: %s" % ", ".join(list(MINIMIZATION_ALGORITHMS)))
        
    algorithm = MINIMIZATION_ALGORITHMS[algorithm]

    return algorithm["fn"](model)