import os.path as osp
import random
import warnings

from dgemm import __name__ as NAME
from dgemm.config import DEFAULT

from bpyutils.util.ml      import get_data_dir
from bpyutils.util.types   import lmap, lfilter
from bpyutils.util.string  import get_random_str
from bpyutils.util.array   import flatten
from bpyutils.util._csv    import (
    write as write_csv
)
from bpyutils import log

import cobra
import pandas as pd
from tqdm import tqdm

from dgemm import settings
from dgemm.model.minimize import minimize_model

from cobra.io.web import load_model as load_gemm
from cobra.io import read_sbml_model, write_sbml_model

warnings.filterwarnings("ignore")

logger = log.get_logger(name = NAME)

cobra_config = cobra.Configuration()

CSV_HEADER_PRE  = []
CSV_HEADER_POST = []

MINIMUM_LOWER_BOUND = -1000
MAXIMUM_UPPER_BOUND =  1000

MAXIMUM_KNOCKOUTS   = 3

def get_random_model_object_ko_sample(model, type_, n = MAXIMUM_KNOCKOUTS, exclude = None):
    exclude = exclude or []

    objekt = getattr(model, type_)
    objekt = lfilter(lambda x: x.id not in exclude, objekt)
    
    n      = max(1, min(n, len(objekt)))

    rand_n = random.randint(1, n)
    sample = random.sample(objekt, rand_n)

    return sample

def knock_out_random_genes(model, output, exclude_rxns = None):
    ko_genes = get_random_model_object_ko_sample(model, "genes", n = MAXIMUM_KNOCKOUTS)

    with model:
        for gene in ko_genes:
            gene.knock_out()
        return optimize_model_and_save(model, output)

def knock_out_random_reactions(model, output, exclude_rxns = None):
    ko_reactions = get_random_model_object_ko_sample(model, "reactions", n = MAXIMUM_KNOCKOUTS, exclude = exclude_rxns)
    
    with model:
        for reaction in ko_reactions:
            reaction.knock_out()
        return optimize_model_and_save(model, output)

def change_random_reaction_bounds(model, output, exclude_rxns = None):
    n = random.randint(1, len(model.reactions))
    random_reactions = get_random_model_object_ko_sample(model, "reactions", n = n, exclude = exclude_rxns)

    with model:
        for reaction in random_reactions:
            if reaction.reversibility:
                reaction.lower_bound = random.uniform(MINIMUM_LOWER_BOUND, 0)
                reaction.upper_bound = random.uniform(0, MAXIMUM_UPPER_BOUND)
            else:
                reaction.upper_bound = random.uniform(reaction.lower_bound, MAXIMUM_UPPER_BOUND)
        return optimize_model_and_save(model, output)

def optimize_model_and_save(model, output, **kwargs):
    solution = model.optimize()
    success  = False

    if solution.status == 'optimal':
        fluxes = list(solution.fluxes)

        row  = flatten(lmap(lambda x: x.bounds, model.reactions))
        row += fluxes

        write_csv(output, row, mode = "a+")

        success = True

    return success

def mutate_model_and_save(strategy, model, output, exclude_rxns = None):
    return strategy(model, output, exclude_rxns = exclude_rxns)

STRATEGIES = [
    knock_out_random_genes,
    knock_out_random_reactions,
    change_random_reaction_bounds
]

def _mutate_step(model, output, exclude_rxns = None):
    strategy = random.choice(STRATEGIES)
    return mutate_model_and_save(strategy, model, output, exclude_rxns = exclude_rxns)

def generate_flux_data(model_id, **kwargs):
    jobs           = kwargs.get("jobs", settings.get("jobs"))
    data_dir       = get_data_dir(NAME, kwargs.get("data_dir"))
    n_data_points  = kwargs.get("n_data_points", DEFAULT["n_flux_data_points"])
    min_model      = kwargs.get("minimize_model", DEFAULT["minimize_model"])

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
            logger.info("Minimizing model: %s" % name)

            min_rxns, minimized_model = minimize_model(model, jobs = jobs)
            write_sbml_model(minimized_model, osp.join(data_dir, "%s_minimized.xml" % name))
    
    name = model.id or model.name or get_random_str()
    output_csv = osp.join(data_dir, "%s.csv" % name)

    reactions = [r for r in model.reactions if r.id not in min_rxns]

    if not osp.exists(output_csv):
        logger.info("Creating output CSV file at path: %s" % output_csv)

        reaction_columns = flatten(
            lmap(lambda x: [x.id + "_lb", x.id + "_ub"], reactions)
        )

        reaction_flux_columns = lmap(lambda x: "%s_flux" % x.id, reactions)
        header = CSV_HEADER_PRE + reaction_columns + reaction_flux_columns + CSV_HEADER_POST
        write_csv(output_csv, header)

        logger.success("Created output CSV file at path: %s" % output_csv)
    else:
        logger.warning("Output CSV file already exists at path: %s" % output_csv)

    with tqdm(total = n_data_points, desc = "Generating Flux Data (%s)" % model.id) as pbar:
        i = 0
        while i < n_data_points:
            success = _mutate_step(model, output_csv, exclude_rxns = min_rxns)

            if success:
                i += 1
                pbar.update(1)

    logger.success("Generated flux data for model: %s" % model_id)
    logger.success("Currently, %s flux data points are available." % len(pd.read_csv(output_csv)))