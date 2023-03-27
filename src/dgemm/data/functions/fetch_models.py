import os.path as osp

from dgemm.config import DEFAULT
from dgemm import __name__ as NAME
from dgemm.data.functions.generate_flux_data import generate_flux_data

from upyog.util.ml      import get_data_dir
from upyog.util.types   import build_fn
from upyog.util.types   import lmap
from upyog.util.string  import strip
from upyog.const        import CPU_COUNT
from upyog              import parallel, log

import cobra
from cobra.io.web.load import DEFAULT_REPOSITORIES as MODEL_REPOSITORIES

logger = log.get_logger(name = NAME)

cobra_config = cobra.Configuration()

def get_all_model_ids():
    model_ids = []
    for repository in MODEL_REPOSITORIES:
        if hasattr(repository, "get_ids"):
            logger.info("Fetching model ids from repository: %s", repository)
            model_ids.extend(repository.get_ids())
    return model_ids

def download_model(model_id, data_dir = None,
    flux_data_dir = None,
    n_data_points = DEFAULT["n_flux_data_points"]
):
    data_dir = get_data_dir(NAME, data_dir = data_dir)
    flux_data_dir = flux_data_dir or data_dir

    generate_flux_data(model_id, data_dir = flux_data_dir, n_data_points = n_data_points)

def fetch_models(data_dir = None, *args, **kwargs):
    logger.info("Fetching Models...")

    jobs = kwargs.get("jobs", CPU_COUNT)
    flux_data_dir = kwargs.get("flux_data_dir", data_dir)
    n_data_points = kwargs.get("n_data_points", DEFAULT["n_flux_data_points"])

    model_id = kwargs.get("model_id", DEFAULT["model_id"])

    if model_id == "all":
        model_ids = get_all_model_ids()
    else:
        model_ids = lmap(strip, model_id.split(","))
    
    with parallel.no_daemon_pool(processes = jobs) as pool:
        function_ = build_fn(download_model, data_dir = data_dir,
            flux_data_dir = flux_data_dir,
            n_data_points = n_data_points)

        for _ in pool.map(function_, model_ids):
            pass