import os.path as osp
import random
import warnings

from dgemm import __name__ as NAME
from dgemm.config import DEFAULT

from upyog.util.ml      import get_data_dir
from upyog.util.types   import lmap, lfilter
from upyog.util.string  import get_random_str
from upyog.util.array   import flatten
from upyog.util._csv    import (
    write as write_csv
)
from upyog import log

import cobra
import pandas as pd
from tqdm import tqdm

from dgemm import settings
from dgemm.model.minimize import minimize_model
from dgemm.data.util import get_random_model_object_sample

from cobra.io.web import load_model as load_gemm
from cobra.io import read_sbml_model, write_sbml_model

warnings.filterwarnings("ignore")

logger = log.get_logger(name = NAME)

cobra_config = cobra.Configuration()

CSV_HEADER_PRE  = []
CSV_HEADER_POST = []

MINIMUM_LOWER_BOUND = -1000
MAXIMUM_UPPER_BOUND =  1000

def knock_out_random_genes(model, output, exclude_rxns = None):
    ko_genes = get_random_model_object_sample(model, "genes")

    with model:
        for gene in ko_genes:
            gene.knock_out()
        return optimize_model_and_save(model, output, strategy = "knock_out_random_genes")

def knock_out_random_reactions(model, output, exclude_rxns = None):
    ko_reactions = get_random_model_object_sample(model, "reactions", exclude = exclude_rxns)
    
    with model:
        for reaction in ko_reactions:
            reaction.knock_out()
        return optimize_model_and_save(model, output, strategy = "knock_out_random_reactions")

def change_random_reaction_bounds(model, output, exclude_rxns = None):
    n = random.randint(1, len(model.reactions))
    random_reactions = get_random_model_object_sample(model, "reactions", n = n, exclude = exclude_rxns)

    with model:
        for reaction in random_reactions:
            if reaction.reversibility:
                reaction.lower_bound = random.uniform(MINIMUM_LOWER_BOUND, 0)
                reaction.upper_bound = random.uniform(0, MAXIMUM_UPPER_BOUND)
            else:
                reaction.upper_bound = random.uniform(reaction.lower_bound, MAXIMUM_UPPER_BOUND)
        return optimize_model_and_save(model, output, strategy = "change_random_reaction_bounds")

STRATEGY_CODE = {
    "knock_out_random_genes"          : 0,
    "knock_out_random_reactions"      : 1,
    "change_random_reaction_bounds"   : 2
}

def optimize_model_and_save(model, output, strategy, **kwargs):
    solution = model.optimize()
    success  = False

    if solution.status == 'optimal':
        fluxes = list(solution.fluxes)

        row  = flatten(lmap(lambda x: x.bounds, model.reactions))
        row += [STRATEGY_CODE[strategy]]
        row += fluxes

        write_csv(output, row, mode = "a+")

        success = True

    return success

def mutate_model_and_save(strategy, model, output, exclude_rxns = None):
    func = strategy["func"]
    return func(model, output, exclude_rxns = exclude_rxns)

STRATEGIES = [{
    "name": "knock_out_random_genes",
    "func": knock_out_random_genes
}, {
    "name": "knock_out_random_reactions",
    "func": knock_out_random_reactions
}, {
    "name": "change_random_reaction_bounds",
    "func": change_random_reaction_bounds
}]

def _mutate_step(model, output, exclude_rxns = None):
    strategy = random.choice(STRATEGIES)
    return mutate_model_and_save(strategy, model, output, exclude_rxns = exclude_rxns)

def generate_flux_data(model_id, **kwargs):
    jobs           = kwargs.get("jobs", settings.get("jobs"))
    data_dir       = get_data_dir(NAME, kwargs.get("data_dir"))
    n_data_points  = kwargs.get("n_data_points", DEFAULT["n_flux_data_points"])
    min_model      = kwargs.get("minimize", DEFAULT["minimize_model"])

    model = None

    logger.info("Generating flux data for model: %s" % model_id)

    logger.info("Loading SBML model: %s" % model_id)

    cobra_config.cache_directory = data_dir

    minimized_model_path = osp.join(data_dir, "%s_minimized.xml" % model_id)
    min_rxns = []

    if osp.exists(minimized_model_path):
        logger.warn("Minimized model already exists: %s" % minimized_model_path)
        
        logger.info("Loading minimized model: %s" % model_id)
        model = read_sbml_model(minimized_model_path)

        min_rxns = lfilter(lambda x: x.lower_bound == 0 and x.upper_bound == 0, model.reactions)
    else:
        model = load_gemm(model_id)
        logger.success("Loaded SBML model: %s" % model_id)
        
        if min_model:
            logger.info("Minimizing model: %s" % model_id)

            min_rxns, minimized_model = minimize_model(model, jobs = jobs)
            write_sbml_model(minimized_model, osp.join(data_dir, "%s_minimized.xml" % model_id))

    name = model.id or model.name or get_random_str()
    
    reactions  = [r for r in model.reactions if r.id not in min_rxns]
    
    output_csv = osp.join(data_dir, "%s.csv" % name)

    if not osp.exists(output_csv):
        logger.info("Creating output CSV file at path: %s" % output_csv)

        reaction_columns = flatten(
            lmap(lambda x: [x.id + "_lb", x.id + "_ub"], reactions)
        )

        reaction_flux_columns = lmap(lambda x: "%s_flux" % x.id, reactions)
        header = CSV_HEADER_PRE + reaction_columns + ["mutation_strategy"] + reaction_flux_columns + CSV_HEADER_POST
        write_csv(output_csv, header)

        logger.success("Created output CSV file at path: %s" % output_csv)
    else:
        logger.warning("Output CSV file already exists at path: %s" % output_csv)

    stats = { "infeasible": 0 }

    with tqdm(total = n_data_points, desc = "Generating Flux Data (%s)" % model.id) as pbar:
        i = 0
        while i < n_data_points:
            success = _mutate_step(model, output_csv, exclude_rxns = min_rxns)

            if success:
                i += 1
                pbar.update(1)
            else:
                stats["infeasible"] += 1

    logger.success("Generated flux data for model: %s" % model_id)
    logger.success("Currently, %s flux data points are available." % len(pd.read_csv(output_csv)))

    return model, stats