from upyog._compat import iteritems

from cobra.util.solver   import linear_reaction_coefficients
from cobra.flux_analysis import flux_variability_analysis
from cobra.io.web.load   import load_model as load_gemm

def find_reaction(model, like, raise_err = True):
    reactions = model.reactions.query(lambda x: like in (x.id or x.name).lower())

    if len(reactions) == 0 and raise_err:
        raise ValueError("Reaction like %s not found." % like)

    return reactions

def find_mle_reactions(model, cutoff = 1e-6, zero_flux_rxns = None):
    tolerance = model.tolerance or cutoff
    final_ids = []
    
    with model:
        objective_value = model.slim_optimize()

        reactions = linear_reaction_coefficients(model)

        for reaction, coeff in iteritems(reactions):
            reaction = model.reactions.get_by_id(reaction.id)
            reaction.lower_bound = objective_value

        fva_solution = flux_variability_analysis(model)

        check_ids = []

        for index, row in fva_solution.iterrows():
            minimum, maximum = row["minimum"], row["maximum"]
            check = max(abs(minimum), abs(maximum)) < tolerance
            
            if check:
                reaction = model.reactions.get_by_id(index)
                check_ids.append(reaction.id)

        final_ids = check_ids

        if zero_flux_rxns:
            zero_flux_rxn_ids = [r.id for r in zero_flux_rxns]
            final_ids = list(set(check_ids) - set(zero_flux_rxn_ids))

    return model.reactions.get_by_any(final_ids)

def normalize_model(m):
    if isinstance(m, str):
        m = load_gemm(m)
    return m