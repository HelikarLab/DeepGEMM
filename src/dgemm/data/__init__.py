import os.path as osp

from dgemm.config import DEFAULT
from dgemm import __name__ as NAME

from upyog.util.ml      import get_data_dir
from upyog.util.system  import (
    get_files
)
from upyog import parallel, log

from dgemm.data.functions import (
    fetch_models,
    generate_flux_data
)
from dgemm import settings

logger = log.get_logger()

def generate_data(data_dir = None, check = False, *args, **kwargs):
    jobs     = kwargs.get("jobs", settings.get("jobs"))
    data_dir = get_data_dir(NAME, data_dir = data_dir)
    # TODO: Generate Data
    files    = get_files(data_dir, "*.gz")

    if check:
        files = (osp.join(data_dir, "%s.xml.gz" % DEFAULT["bigg_model_id"]),)
    
    with parallel.no_daemon_pool(processes = jobs) as pool:
        pool.lmap(generate_flux_data, files)

def get_data(data_dir = None, check = False, *args, **kwargs):
    data_dir = get_data_dir(NAME, data_dir)

    logger.info("Fetching models...")
    fetch_models(data_dir = data_dir, check = check, *args, **kwargs)

    # logger.info("Generating online data...")
    # generate_data(data_dir = data_dir, check = check, *args, **kwargs)

def preprocess_data(data_dir = None, check = False, *args, **kwargs):
    data_dir = get_data_dir(NAME, data_dir = data_dir)
    # do something ...