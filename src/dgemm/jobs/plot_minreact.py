import os.path as osp
import time

from dgemm.model.minimize import minimize_model
from dgemm.jobs.helper import perform_on_models, plot_3d_graph
from dgemm.config import PATH

from bpyutils import log
from bpyutils.util._json import JSONLogger
from bpyutils.const import CPU_COUNT

import cobra

logger = log.get_logger(__name__)
KIND   = "minreact"

stats_logger = JSONLogger("%s.json" % KIND)

def min_model(model_id, jobs = None):
    logger.info("Generating data for model %s...", model_id)

    try:
        start  = time.time()
        min_rxns, model, minimized = minimize_model(model_id, jobs = jobs)
        end    = time.time()

        path_model = osp.join(PATH["CACHE"], f"{model_id}_minimized.json")
        cobra.io.write_sbml_model(minimized, path_model)

        n_rxns = len(model.reactions)

        stats = {
            "metabolites":   len(model.metabolites),
            "reactions":     n_rxns,
            "min_reactions": n_rxns - len(min_rxns),
            "genes":         len(model.genes),
            "time":          (end - start)
        }

        stats_logger[model_id] = stats
        stats_logger.save()
        
        logger.info("Plotting graph...")
        plot_3d_graph(stats_logger.store, "metabolites", "reactions", "min_reactions", prefix = KIND, suffix = "minimized reactions")
        plot_3d_graph(stats_logger.store, "metabolites", "reactions", "time", prefix = KIND, suffix = "time")
    except Exception as e:
        logger.error(f"Failed to generate data for model {model_id}: {e}")

if __name__ == "__main__":
    exclude = list(stats_logger.store)
    perform_on_models(min_model, exclude = exclude, load = False, shuffle = True, jobs = CPU_COUNT)