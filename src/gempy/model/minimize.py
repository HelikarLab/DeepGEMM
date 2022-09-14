from bpyutils.util.array import squash
from bpyutils._compat import iteritems

from cobra.flux_analysis import (
    pfba,
    find_blocked_reactions
)
from cobra.flux_analysis.helpers import (
    normalize_cutoff
)
from cobra.util.solver import (
    linear_reaction_coefficients
)

from gempy.model.util import (
    find_reaction,
    find_mle_reactions
)
from gempy.const import DEFAULT

def minreact(model,
    retain      = DEFAULT["min_react_reaction_retention_list"],
    zero_cutoff = 1.0
):
    model_copy = model.copy()

    zero_cutoff = normalize_cutoff(model_copy, zero_cutoff)

    objective_reactions = linear_reaction_coefficients(model_copy)

    objective_value = model_copy.slim_optimize()

    # with model:
    for objective_reaction, coeff in iteritems(objective_reactions):
        objective_reaction.lower_bound = growth_rate_cutoff * objective_value
        
    pfba_solution = pfba(model_copy)

    remove_reaction_ids = find_blocked_reactions(model)
    
    zero_flux_rxn_ids = []
        
    for i, flux in enumerate(pfba_solution.fluxes):
        if flux == 0:
            reaction = model.reactions[i]
            zero_flux_rxn_ids.append(reaction.id)
            
        remove_reaction_ids += zero_flux_rxn_ids
        remove_reaction_ids += find_mle_reactions(model, zero_flux_rxns = model.reactions.get_by_any(zero_flux_rxn_ids))

        model.remove_reactions(remove_reaction_ids)

    return model_copy

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