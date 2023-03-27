import os.path as osp
import time

from dgemm.data.functions.generate_flux_data import generate_flux_data
from dgemm.jobs.helper import perform_on_models, plot_graph, DEFAULT_ARTIFACTS_DIR

from upyog import log, parallel
from upyog.util._json import JSONLogger
from upyog.util._dict import merge_dict
from upyog.util.types import build_fn
from upyog.const import CPU_COUNT

logger = log.get_logger(__name__)
KIND   = "datagen"

def gen_data(model_id, jobs = None, **kwargs):
    artifacts_dir = kwargs.get("artifacts_dir", DEFAULT_ARTIFACTS_DIR)
    logger_fpath  = kwargs.get("logger_fpath", osp.join(artifacts_dir, "%s.json" % KIND))
    n_data_points = kwargs.get("n_data_points", 1000)

    stats_logger  = JSONLogger(logger_fpath)

    logger.info("Generating data for model %s...", model_id)

    try:
        start = time.time()
        model, stats = generate_flux_data(model_id, n_data_points = n_data_points, jobs = jobs)
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
        plot_graph(stats_logger.store, "reactions", "genes", "infeasible", prefix = KIND, suffix = "infeasible", dir_path = artifacts_dir)
        plot_graph(stats_logger.store, "metabolites", "reactions", "time", prefix = KIND, suffix = "time", dir_path = artifacts_dir)
    except Exception as e:
        logger.error(f"Failed to generate data for model {model_id}: {e}")

def run(*args, **kwargs):
    artifacts_dir = kwargs.get("artifacts_dir", DEFAULT_ARTIFACTS_DIR)
    include = kwargs.get("model_id", None)
    jobs = kwargs.get("jobs", CPU_COUNT)
    n_data_points = kwargs.get("n_data_points", 1000)

    if include:
        include = include.split(",")

    filename      = osp.join(artifacts_dir, "%s.json" % KIND)
    stats_logger  = JSONLogger(filename)

    if include:
        with parallel.no_daemon_pool(processes = jobs) as pool:
            fn = build_fn(gen_data, artifacts_dir = artifacts_dir, logger_fpath = filename, n_data_points = n_data_points, jobs = jobs)
            list(pool.map(fn, include))
    else:
        exclude = list(stats_logger.store)
        perform_on_models(gen_data, exclude = exclude, load = False, shuffle = True, jobs = CPU_COUNT, kwargs = {
            "artifacts_dir": artifacts_dir,
            "logger_fpath": filename
        })