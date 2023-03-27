# imports - standard imports
import os.path as osp
import random
import warnings
import time

# imports - data-science imports
import numpy as np
import pandas as pd
import cobra
from cobra import Metabolite, Reaction
from cobra.flux_analysis import gapfill
from cobra.flux_analysis.parsimonious import pfba
from cobra.flux_analysis.variability import find_blocked_reactions, flux_variability_analysis
from cobra.flux_analysis.deletion import single_reaction_deletion
from cobra.flux_analysis.helpers import normalize_cutoff
from cobra.util.solver import set_objective
from cobra.util import linear_reaction_coefficients

# imports - third-party and other utility imports
import tqdm.notebook as tq
from upyog.util._dict import dict_from_list
from upyog.util._csv  import write as write_csv
from upyog.util.types import lmap, lfilter, build_fn
from upyog.util.array import flatten, squash
from upyog.util.imports import import_handler
from upyog._compat import iterkeys, iteritems
from upyog import log, parallel
from upyog.log import get_logger
from upyog.const import CPU_COUNT

from upyog.util.array import squash
from upyog._compat import iteritems
from upyog.log import get_logger

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

from dgemm.model.util import (
    find_reaction,
    find_mle_reactions,
    normalize_model
)
from dgemm.model.optimize import optimize as optimize_model
from dgemm.const import DEFAULT
from dgemm import __name__ as NAME

logger = get_logger(NAME)

BIGG_ID_ATPM = "ATPM"

warnings.filterwarnings('ignore')

cobra_logger = get_logger("cobrapy")
cobra_logger.setLevel(log.DEBUG)

# constants
cobra_config = cobra.Configuration()
cobra_config.tolerance = 1e-6

def _single_syntehtic_lethality_step(rxn, m, cutoff = 0.01, objective_value = None):
    with m as copy:
        rxn = copy.reactions.get_by_id(rxn)
        rxn.knock_out()

        solution = copy.slim_optimize()

        if np.isnan(solution) or solution < (cutoff * objective_value):
            return rxn.id

def single_synthetic_lethality(m, cutoff = 0.01, eliminate = [], tolerance = None, processes = None):
    """
        Get single synthetic lethal reactions.
    """ 
    tolerance = normalize_cutoff(m, tolerance)
    processes = processes or CPU_COUNT
    _log_model(m, "info", "Optimizing (taxicab norm) using Synthetic Lethality...")
    solution  = optimize_model(m, type_ = None, objective_sense = "maximize")
    objective_value = solution.objective_value
    _log_model(m, "info", "Optimized. Objective Value: %s" % objective_value)
    
    fluxes = solution.fluxes

    pruned_rxns = list(
        set(fluxes[abs(fluxes) > tolerance].index) - set(eliminate)
    )
    _log_model(m, "info", "Pruned Reactions: %s" % len(pruned_rxns))

    rxn_ids = []

    # for rxn in tq.tqdm(pruned_rxns, desc = "Checking for objective on knockout"):
    #     _log_model(m, "info", "Checking for objective on knockout: %s" % rxn)
    #     if _single_syntehtic_lethality_step(rxn, m,
    #             cutoff = cutoff, objective_value = objective_value):
    #         rxn_ids.append(rxn)
    # with parallel.no_daemon_pool(processes = processes) as pool:
    #     fn = build_fn(_single_syntehtic_lethality_step,
    #                      m = m, cutoff = cutoff, objective_value = objective_value)
    #     for rxn in tq.tqdm(pool.imap(fn, pruned_rxns), total = len(pruned_rxns), desc = "Synthetic Lethality, checking for objective on knockout..."):
    #         if rxn:
    #             rxn_ids.append(rxn)

    reaction_deletions = single_reaction_deletion(m, processes = processes, reaction_list = pruned_rxns)
    for deletion_data in reaction_deletions.itertuples():
        if np.isnan(deletion_data.growth) or deletion_data.growth < (cutoff * objective_value):
            rxn_ids.append(squash(list(deletion_data.ids)))
    
    return rxn_ids

def convert_to_irreversible(m):
    irr = m.copy()
    
    add_reactions = []
    coefficients  = {}
    
    for reaction in irr.reactions:
        if reaction.lower_bound < 0:
            reverse_reaction = Reaction("%s_reverse" % reaction.id)
            reverse_reaction.lower_bound = max(0, -reaction.upper_bound)
            reverse_reaction.upper_bound = max(0, -reaction.lower_bound)
            
            coefficients[reverse_reaction] = reaction.objective_coefficient * -1
            
            reaction.lower_bound = max(0, reaction.lower_bound)
            reaction.upper_bound = max(0, reaction.upper_bound)

            reaction.notes["reflection"] = reverse_reaction.id
            reverse_reaction.notes["reflection"] = reaction.id
            
            reverse_reaction_metabolites = \
                { k: v * -1 for k, v in iteritems(reaction.metabolites) }
            
            reverse_reaction.add_metabolites(reverse_reaction_metabolites)
            
            reverse_reaction._model = reaction._model
            reverse_reaction._genes = reaction._genes
            
            for gene in reaction._genes:
                gene._reaction.add(reverse_reaction)
                
            reverse_reaction.subsystem = reaction.subsystem
            reverse_reaction.gene_reaction_rule = reaction.gene_reaction_rule
            
            add_reactions.append(reverse_reaction)
            
    irr.add_reactions(add_reactions)
    set_objective(irr, coefficients, additive = True)

    return irr

def minimize_flux(m):
    model_irr = convert_to_irreversible(m)
    name = 'min_flux'
    
    metabolite = Metabolite(name)
    
    for reaction in model_irr.reactions:
        reaction.add_metabolites({
            metabolite: 1
        })

    reaction = Reaction(name)
    reaction.lower_bound = 0
    reaction.upper_bound = cobra_config.upper_bound
    reaction.add_metabolites({ metabolite: -1 })
    model_irr.add_reactions([reaction])

    model_irr.objective = name
    
    solution = model_irr.optimize(objective_sense = "minimize", raise_error = True)
    objective_value = solution.objective_value
    
    _log_model(m, "info", "Objective Value (irr): %s" % objective_value)
    
    _log_model(m, "info", "number of metabolites, reactions and genes (irr): %s, %s, %s" \
        % (len(model_irr.metabolites), len(model_irr.reactions), len(model_irr.genes)))
        
    return objective_value, model_irr

def get_fva_mapped_rxns(m, fn, *args, **kwargs):
    kwargs["processes"] = kwargs.get("processes", CPU_COUNT)
    kwargs["loopless"]  = kwargs.get("loopless", False)

    _log_model(m, "info", "Using %s jobs for FVA." % kwargs["processes"])
    
    fva_solution = flux_variability_analysis(m, *args, **kwargs)
    
    rxns = []
    
    for i in range(len(fva_solution)):
        min_flux, max_flux = fva_solution["minimum"][i], fva_solution["maximum"][i]
        value = fn(min_flux, max_flux)
        
        if value:
            reaction = m.reactions[i]
            rxns.append(reaction.id)

    return rxns

def delete_reactions(m, rxns, remove_orphans = False):
    for reaction_id in rxns:
        reaction = m.reactions.get_by_id(reaction_id)
        reaction.delete(remove_orphans = remove_orphans)
    return m

def cut_off_obj_rxn_lb(m, growth_rate_cutoff = 1):
    # perform fba to obtain maximum growth rate of objective reaction
    _log_model(m, "info", "Performing FBA to obtain maximum growth rate of objective reaction...")
    objective_reactions = list(linear_reaction_coefficients(m))
    objective_value = m.slim_optimize()
    
    # set the lower_bound of the objective reaction to growth_rate_cutoff * objective_value
    objective_value_cutoff = growth_rate_cutoff * objective_value
    for objective_reaction in objective_reactions:
        objective_reaction.lower_bound = objective_value_cutoff
    
    _log_model(m, "info", "wild-type growth rate: %.4f" % objective_value_cutoff)

    return m, objective_value_cutoff

def _log_model(m, type_, msg):
    log_type = getattr(logger, type_)
    log_type("[%s] %s" % (m.id, msg))

def minreact(m, growth_rate_cutoff = 1, tolerance = None, maintain = (BIGG_ID_ATPM,), jobs = CPU_COUNT):
    """
    Description:
        MinReact algorithm
    
    Parameters:
        growth_rate_cutoff: minimum cut-off of wild-type growth the resulting network 
            should retain. (default: 1)
        tolerance: minimum flux a reaction should possess to be considered active 
            (default: `model.tolerance`)
        maintain: a list of reactions to be preserved (default: "ATPM")
        
    Returns:
        model: minimized model
    """
    m = normalize_model(m)

    minimized = m.copy()
    tolerance = normalize_cutoff(m, tolerance)

    maintain_map = dict_from_list(maintain)
    
    _log_model(m, "info", "Using tolerance: %s" % tolerance)
    
    minimized, objective_value_cutoff = cut_off_obj_rxn_lb(minimized, growth_rate_cutoff)
    
    # identifiy reaction classes - blocked, zero flux and mle (metabolically 
    # less efficient) reactions
    remove_reactions = []
    pfba_opt_rxns = []
    
    with minimized as minimized_copy:
        # find blocked reactions
        _log_model(m, "info", "Finding and deleting blocked reactions...")
        blocked_reactions = find_blocked_reactions(minimized_copy, open_exchanges = True,
                                                   processes = jobs)
        _log_model(m, "success", "Found %s blocked reactions." % len(blocked_reactions))
        remove_reactions += blocked_reactions

        minimized_copy = delete_reactions(minimized_copy, blocked_reactions)

        _log_model(m, "info", "Currently %s reactions within model." % len(minimized_copy.reactions))

        _log_model(m, "info", "Fetching single synthetic lethal reactions...")
        pfba_essential_rxns = single_synthetic_lethality(minimized_copy, eliminate = maintain, processes = jobs)
        _log_model(m, "success", "Found %s pfba essential reactions." % len(pfba_essential_rxns))
        
        # find zero flux reactions
        zero_flux_rxns = []

        _log_model(m, "info", "Finding zero-flux reactions with FVA...")
        zero_flux_rxns = get_fva_mapped_rxns(minimized_copy,
            fn = lambda min_f, max_f: abs(min_f) < tolerance and abs(max_f) < tolerance,
            fraction_of_optimum = tolerance, processes = jobs)
        _log_model(m, "success", "Found %s zero-flux reactions with FVA" % len(zero_flux_rxns))

        minimized_copy = delete_reactions(minimized_copy, zero_flux_rxns)
            
        remove_reactions += zero_flux_rxns
    
        # find metabolically less efficient (mle) reactions
        _log_model(m, "info", "Finding metabolically less efficient reactions...")
        zero_flux_rxn_map = dict_from_list(zero_flux_rxns)

        minimized_copy, _ = cut_off_obj_rxn_lb(minimized_copy, growth_rate_cutoff)
    
        fva_mapped_rxns = get_fva_mapped_rxns(minimized_copy,
            lambda min_f, max_f: max(abs(min_f), abs(max_f)) < tolerance
        )
        _log_model(m, "info", "Found %s reactions with flux < %s" % (len(fva_mapped_rxns), tolerance))
        mle_rxns = [rxn for rxn in fva_mapped_rxns if rxn not in zero_flux_rxn_map]
        
        _log_model(m, "info", "Found %s MLE reactions." % len(mle_rxns))
        
        _log_model(m, "info", "Currently %s metabolites, %s reactions in model." % 
                    (len(minimized_copy.metabolites),
                     len(minimized_copy.reactions)))
        
        # find pFBA optimal reactions
        _log_model(m, "info", "Minimizing flux...")
        min_flux_value, irr_model = minimize_flux(minimized_copy)
        _log_model(m, "info", "Minimized flux: %s" % min_flux_value)
        irr_model.reactions.min_flux.lower_bound = min_flux_value
        irr_model.reactions.min_flux.upper_bound = min_flux_value

        fva_irr_rxns = get_fva_mapped_rxns(irr_model,
            lambda min_f, max_f: (abs(min_f) + abs(max_f)) >= tolerance
        )

        _log_model(m, "info", "FVA irr n reactions: %s" % len(fva_irr_rxns))

        _log_model(m, "info", "Finding optimal reactions...")
        pfba_opt_rxns = fva_irr_rxns

        # pfba_opt_rxns = lfilter(lambda x: not x.endswith("_reverse"), list(pfba_opt_rxns))
        # _log_model(m, "info", "Found %s pFBA optimal reactions (after removing reverse reactions)." % len(pfba_opt_rxns))
        pfba_opt_rxns = list(set(lmap(lambda x: x.replace("_reverse", ""), pfba_opt_rxns)))
        
        pfba_opt_rxns = np.setdiff1d(pfba_opt_rxns, mle_rxns)
        pfba_opt_rxns = np.setdiff1d(pfba_opt_rxns, pfba_essential_rxns)
        
        _log_model(m, "info", "Found %s pfba opt reactions." % len(pfba_opt_rxns))
        _log_model(m, "info", "pfba opt reactions are: %s" % pfba_opt_rxns)

        remove_reactions += mle_rxns
        
    # remove the reaction classes from the model
    _log_model(m, "info", "Removing %s reaction classes from model..." % len(remove_reactions))
    for reaction in remove_reactions:
        minimized.reactions.get_by_id(reaction).knock_out()

    pfba_opt_rxns = lfilter(lambda x: x in minimized.reactions, pfba_opt_rxns)

    pfba_opt_rxns = np.setdiff1d(pfba_opt_rxns, maintain)
    _log_model(m, "info", "Using %s pfba opt reactions." % len(pfba_opt_rxns))

    o_rxn = []
    o_rxn_sum = []

    for o_idx, opt_rxn_id in enumerate(pfba_opt_rxns):
        with minimized as min_copy:
            reaction = min_copy.reactions.get_by_id(opt_rxn_id)
            reaction.knock_out()
            
            solution = min_copy.optimize(objective_sense = 'maximize')
            fluxes   = solution.fluxes
            objective_value = solution.objective_value
        
            non_zero_fluxes = fluxes \
                .where(lambda x: x != 0)
                
            non_zero_flux_maintain_map = lfilter(
                lambda x: x in maintain_map, non_zero_fluxes.index
            )

            if objective_value > 0 and len(non_zero_flux_maintain_map) == len(maintain_map):
                abs_fluxes = fluxes.abs() > tolerance

                abs_gt_tol_rxns = list(fluxes \
                    .where(lambda x: abs(x) > tolerance)
                    .index)
                abs_gt_tol_rxn_map = dict_from_list(abs_gt_tol_rxns)
                
                for reaction in min_copy.reactions:
                    if reaction.id not in abs_gt_tol_rxn_map:
                        reaction.knock_out()
                    
                sol = min_copy.optimize(objective_sense = "maximize")
                if sol.status == "optimal" and \
                    round(sol.objective_value, 4) >= round(objective_value_cutoff, 4):
                    o_rxn.append(abs_fluxes)
                    o_rxn_sum.append(sum(abs_fluxes))

    min_rxns = []
    compressed = m.copy()

    if o_rxn_sum:
        min_react = np.amin(o_rxn_sum)
        _log_model(m, "info", "MinReact: %s" % min_react)

        min_react_idx  = np.where(o_rxn_sum == min_react)[0][0]
        min_react_rxns = o_rxn[min_react_idx]

        for i, reaction in enumerate(compressed.reactions):
            if not min_react_rxns[i]:
                min_rxns.append(reaction.id)
                reaction.knock_out()

        _log_model(m, "info", "Removed %s reactions." % len(min_rxns))
    else:
        _log_model(m, "info", "No reactions removed.")

    return min_rxns, m, compressed

MINIMIZATION_ALGORITHMS = {
    "minreact": {
        "fn": minreact
    }
}

def minimize_model(model, algorithm = "minreact", **kwargs):
    if algorithm not in MINIMIZATION_ALGORITHMS:
        raise ValueError("Not a valid algorithm, accepted are: %s" % ", ".join(list(MINIMIZATION_ALGORITHMS)))
        
    algorithm = MINIMIZATION_ALGORITHMS[algorithm]

    return algorithm["fn"](model, **kwargs)