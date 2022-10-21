import time

from dgemm.data.functions.generate_flux_data import generate_flux_data
from dgemm.jobs.helper import perform_on_models, plot_3d_graph

from bpyutils import log
from bpyutils.util._json import JSONLogger
from bpyutils.util._dict import merge_dict
from bpyutils.const import CPU_COUNT

logger = log.get_logger(__name__)
KIND   = "datagen"

stats_logger = JSONLogger("%s.json" % KIND)

def gen_data(model_id, jobs = None):
    logger.info("Generating data for model %s...", model_id)

    try:
        start = time.time()
        model, stats = generate_flux_data(model_id, n_data_points = 1000, jobs = jobs)
        end   = time.time()

        stats = merge_dict(stats, {
            "metabolites": len(model.metabolites),
            "reactions":   len(model.reactions),
            "genes":       len(model.genes),
            "time":        (end - start)
        })

        stats_logger[model_id] = stats
        stats_logger.save()
        
        logger.info("Plotting graph...")
        plot_3d_graph(stats_logger.store, "reactions", "genes", "infeasible", prefix = KIND, suffix = "infeasible")
        plot_3d_graph(stats_logger.store, "metabolites", "reactions", "time", prefix = KIND, suffix = "time")
    except Exception as e:
        logger.error(f"Failed to generate data for model {model_id}: {e}")

def run(*args, **kwargs):
    exclude = list(stats_logger.store)
    perform_on_models(gen_data, exclude = exclude, load = False, shuffle = True, jobs = CPU_COUNT)