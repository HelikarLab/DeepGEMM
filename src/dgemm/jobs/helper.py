import random
import os.path as osp

from cobra.io.web.load import DEFAULT_REPOSITORIES, load_model

from bpyutils.util.types import build_fn
from bpyutils.util.array import flatten, sequencify
from bpyutils.util.system import makedirs
from bpyutils import parallel, log
from bpyutils._compat import iteritems

from tqdm.auto import tqdm as tq

logger = log.get_logger(__name__)

DEFAULT_ARTIFACTS_DIR = osp.join(osp.expanduser("~"), "dgemm-artifacts")
makedirs(DEFAULT_ARTIFACTS_DIR, exist_ok = True)

def plot_3d_graph(stats, x_label, y_label, z_label, prefix, suffix = "", fontsize_label = 16, dir_path = ""):
    import matplotlib.pyplot as pplt
    from mpl_toolkits import mplot3d

    fig = pplt.figure(figsize = (20, 16))
    ax  = pplt.axes(projection = "3d")

    labels = [x_label, y_label, z_label]

    x, y, z = [], [], []
    for _, item in iteritems(stats):
        x.append(item[x_label])
        y.append(item[y_label])
        z.append(item[z_label])

    ax.scatter(x, y, z, c = z, cmap='viridis', linewidth=0.5)

    # for model_id, item in iteritems(stats_logger.store):
    #     x, y, z = item[x_label], item[y_label], item[z_label]
    #     ax.text3D(x, y, z, model_id, fontsize = 8,
    #         horizontalalignment = "left", verticalalignment = "bottom",
    #         transform = ax.transData, zdir = "z")

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    filename = "%s%s.png" % (prefix, ("-" + suffix) if suffix else "")
    filepath = osp.join(dir_path, filename)

    pplt.savefig(filepath)

    pplt.close()

def _load_model(model_id, repo = None):
    repo = sequencify(repo or DEFAULT_REPOSITORIES)

    try:
        model = load_model(model_id, repositories = repo)
        return model
    except Exception as e:
        logger.error(f"Failed to load {model_id}: {e}")

def get_models(repo, exclude = None, load = False, shuffle = False, jobs = None):
    models = []

    if hasattr(repo, "get_ids"):
        models = repo.get_ids()
        
        if exclude:
            models = list(set(models) - set(exclude))

        if load:
            with parallel.no_daemon_pool(processes = jobs) as pool:
                fn = build_fn(_load_model, repo = repo)
                models = list(tq(pool.imap(fn, models), total=len(models), desc=f"Loading models from {repo.name}..."))
    else:
        logger.warning(f"Repository {repo.name} does not have a get_ids method.")

    if shuffle:
        random.shuffle(models)

    return models

def perform_on_models(to_perform, exclude = None, load = False, shuffle = False, jobs = None, kwargs = None):
    kwargs = kwargs or {}

    with parallel.no_daemon_pool() as pool:
        fn = build_fn(get_models, exclude = exclude, load = load, shuffle = shuffle, jobs = jobs)
        models = flatten(pool.map(fn, DEFAULT_REPOSITORIES))
        
        logger.info(f"Found {len(models)} models.")

        with parallel.no_daemon_pool(processes = jobs) as pool:
            fn = build_fn(to_perform, jobs = jobs, **kwargs)
            list(tq(pool.imap(fn, models), total=len(models), desc="Performing on models..."))