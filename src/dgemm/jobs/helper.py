import random
import os.path as osp

from cobra.io.web.load import DEFAULT_REPOSITORIES, load_model

from bpyutils.util.types import build_fn
from bpyutils.util.array import flatten, sequencify
from bpyutils.util.system import makedirs
from bpyutils.util.string import get_random_str
from bpyutils import parallel, log
from bpyutils._compat import iteritems

from tqdm.auto import tqdm as tq

logger = log.get_logger(__name__)

DEFAULT_ARTIFACTS_DIR = osp.join(osp.expanduser("~"), "dgemm-artifacts")
makedirs(DEFAULT_ARTIFACTS_DIR, exist_ok = True)

def plot_graph(stats, x_feat, y_feat, z_feat = None, prefix = "", suffix = "", dir_path = "", *args, **kwargs):
    x_label = kwargs.get("x_label", x_feat)
    y_label = kwargs.get("y_label", y_feat)
    z_label = kwargs.get("z_label", z_feat)
    prefix  = prefix or get_random_str()

    marker_size    = kwargs.get("marker_size", 10)
    fontname       = kwargs.get("fontname", None)
    fontsize_label = kwargs.get("fontsize_label", 10)
    title          = kwargs.get("title", None)
    show_text      = kwargs.get("show_text", True)
    z_label_ratio  = kwargs.get("z_label_ratio", 0.5)

    import matplotlib.pyplot as pplt
    from mpl_toolkits import mplot3d

    pplt.style.use("seaborn-v0_8-paper")

    fig = pplt.figure(figsize = (8, 6))
    ax  = pplt.axes(projection = "3d" if z_feat else None)

    labels = [x_label, y_label, z_label]

    x, y, z = [], [], []
    for _, item in iteritems(stats):
        x.append(item[x_feat])
        y.append(item[y_feat])

        if z_feat:
            z.append(item[z_feat])

    if z_feat:
        ax.scatter(x, y, z, c = z, cmap = "viridis", linewidth=0.5, s = marker_size)
    else:
        ax.scatter(x, y, c = "black", s = marker_size)

    if z:
        z_max = max(z)

    if show_text:
        text_kwargs = {
            "fontsize": 6,
            "bbox": {
                "facecolor": "white",
                "edgecolor": "black",
                "boxstyle": "round",
                "alpha": 0.3
            },
            "horizontalalignment": "left",
            "verticalalignment": "bottom",
            "transform": ax.transData
        }
        for model_id, item in iteritems(stats):
            x, y = item[x_feat], item[y_feat]
            
            if z_feat:
                z = item[z_feat]

                if z >= z_label_ratio * z_max:
                    ax.text3D(x, y, z, model_id, zdir = "x", **text_kwargs)
            else:
                ax.text(x, y, model_id, **text_kwargs)

    ax.set_xlabel(labels[0], fontsize = fontsize_label, fontname = fontname)
    ax.set_ylabel(labels[1], fontsize = fontsize_label, fontname = fontname)

    if z_feat:
        ax.set_zlabel(labels[2], fontsize = fontsize_label, fontname = fontname)

    if title is not None:
        ax.set_title(title, fontsize = fontsize_label, fontname = fontname)

    if title:
        ax.set_title(title, fontsize = fontsize_label)

    filename = "%s%s.png" % (prefix, ("-" + suffix) if suffix else "")
    filepath = osp.join(dir_path, filename)

    fig.colorbar(ax.collections[-1], ax = ax, location = "right", shrink = 0.7)

    pplt.tight_layout()
    pplt.savefig(filepath, dpi = 300)

    pplt.close()

def _load_model(model_id, repo = None):
    repo = sequencify(repo or DEFAULT_REPOSITORIES)

    try:
        model = load_model(model_id, repositories = repo)
        return model
    except Exception as e:
        logger.error(f"Failed to load {model_id}: {e}")

def get_models(repo, include = None, exclude = None, load = False, shuffle = False, jobs = None):
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

def perform_on_models(to_perform, include = None, exclude = None, load = False, shuffle = False, jobs = None, kwargs = None):
    kwargs = kwargs or {}

    with parallel.no_daemon_pool() as pool:
        fn = build_fn(get_models, exclude = exclude, load = load, shuffle = shuffle, jobs = jobs)
        models = flatten(pool.map(fn, DEFAULT_REPOSITORIES))
        
        logger.info(f"Found {len(models)} models.")

        with parallel.no_daemon_pool(processes = jobs) as pool:
            fn = build_fn(to_perform, jobs = jobs, **kwargs)
            list(tq(pool.imap(fn, models), total=len(models), desc="Performing on models..."))