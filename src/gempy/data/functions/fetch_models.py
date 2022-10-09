import os.path as osp

from dgemm.config import DEFAULT
from dgemm import __name__ as NAME
from dgemm.data.functions.generate_flux_data import generate_flux_data

from bpyutils.util.ml      import get_data_dir
from bpyutils.util.types   import build_fn
from bpyutils.util.types   import lmap
from bpyutils.util.string  import strip
from bpyutils.const        import CPU_COUNT
from bpyutils              import parallel, log

from bioservices import BiGG

logger = log.get_logger(name = NAME)

def download_bigg_model(model_id, data_dir = None,
    gen_flux_data = False,
    flux_data_dir = None,
    n_data_points = DEFAULT["n_flux_data_points"]
):
    data_dir = get_data_dir(NAME, data_dir = data_dir)
    target   = osp.join(data_dir, "%s.xml.gz" % model_id)
    flux_data_dir = flux_data_dir or data_dir

    if not osp.exists(target):
        logger.info("Downloading BiGG Model %s..." % model_id)

        bigg = BiGG()
        bigg.download(model_id, format_ = "xml", target = target)
    else:
        logger.warn("BiGG Model %s already downloaded." % model_id)

    if gen_flux_data:
        generate_flux_data(target, data_dir = flux_data_dir, n_data_points = n_data_points)

def fetch_bigg_models(data_dir = None, check = False, *args, **kwargs):
    jobs = kwargs.get("jobs", CPU_COUNT)
    flux_data_dir = kwargs.get("flux_data_dir", data_dir)
    gen_flux_data = kwargs.get("gen_flux_data", False),
    n_data_points = kwargs.get("n_data_points", DEFAULT["n_flux_data_points"])

    bigg = BiGG()
    model_ids = lmap(lambda x: x["bigg_id"], bigg.models)

    if check:
        model_ids_arg = kwargs.get("model_id", DEFAULT["bigg_model_id"])
        model_ids_arg = lmap(strip, model_ids_arg.split(","))

        model_ids = model_ids_arg
    
    with parallel.no_daemon_pool(processes = jobs) as pool:
        function_ = build_fn(download_bigg_model, data_dir = data_dir,
            gen_flux_data = gen_flux_data,
            flux_data_dir = flux_data_dir,
            n_data_points = n_data_points)

        for _ in pool.map(function_, model_ids):
            pass

def fetch_models(data_dir = None, check = False, *args, **kwargs):
    logger.info("Fetching models...")
    fetch_bigg_models(data_dir = data_dir, check = check, *args, **kwargs)