# from dgemm.__attr__ import __name__ as NAME
# from dgemm.data import get_data_dir

# def build_model(artifacts_path = None):
#     dropout_rate  = settings.get("dropout_rate")
#     batch_norm    = settings.get("batch_norm")

#     from deeply.model.gan import GAN

#     model = GAN()
#     # do something ...
#     pass

# def train(data_dir = None, artifacts_dir = None, check = False, *args, **kwargs):
#     batch_size    = 1 if check else settings.get("batch_size")
#     learning_rate = settings.get("learning_rate")
#     epochs        = 1 if check else settings.get("epochs")

#     data_dir = get_data_dir(NAME, data_dir)
#     model    = build_model()
    
#     dops.watch(model)

#     # do something ...


# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics    import binary_accuracy

# from deeply.model.gan     import GAN

# from deeply.datasets.util import SPLIT_TYPES
# from deeply.generators    import ImageMaskGenerator
# from deeply.losses        import dice_loss

import os.path as osp
import warnings

from bpyutils.util.system import makepath, get_files
from bpyutils.util.types import build_fn
from bpyutils.util.ml import get_data_dir
from bpyutils.log import get_logger
from bpyutils.const import CPU_COUNT
from bpyutils._compat import iterkeys
from bpyutils import parallel

from cobra.io.web import load_model as load_gemm
from cobra.io import read_sbml_model
from cobra.util import linear_reaction_coefficients
import cobra

# import deeply

from dgemm import settings, __name__ as NAME
from dgemm import settings #, dops
from dgemm.data.util import get_random_model_object_sample

warnings.filterwarnings("ignore")

logger = get_logger(NAME)

cobra_config = cobra.Configuration()

try:
    raise ImportError
    
    import dask.dataframe as dfl
    from dask_ml.model_selection import (
        train_test_split, KFold)
    from dask_ml.linear_model import LinearRegression

    from dask.distributed import Client
    Client()

    logger.info("Using Dask for parallel processing")
except ImportError:
    logger.warn("Dask is not installed, using pandas, scikit-learn instead")

    import pandas as dfl
    from sklearn.model_selection import (
        train_test_split, KFold)

    from sklearn.linear_model import LinearRegression
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.neural_network import MLPRegressor


def build_model(artifacts_path = None):
    encoder_dropout_rate = settings.get("encoder_dropout_rate")
    encoder_batch_norm   = settings.get("encoder_batch_norm")
    decoder_dropout_rate = settings.get("decoder_dropout_rate")
    decoder_batch_norm   = settings.get("decoder_batch_norm")

    gan = deeply.hub("gan", x = 100,
        encoder_dropout_rate = encoder_dropout_rate,
        encoder_batch_norm   = encoder_batch_norm,
        decoder_dropout_rate = decoder_dropout_rate,
        decoder_batch_norm = decoder_batch_norm
    )

    # width, height = settings.get("image_width"), \
    #     settings.get("image_height")

    # unet = UNet(x = width, y = height, channels = 3, n_classes = 1,
    #     final_activation = "sigmoid", batch_norm = batch_norm, 
    #     dropout_rate = dropout_rate, padding = "same",
    #     backbone = "efficient-net-b7", backbone_weights = "imagenet")

    # # unet.
    
    if artifacts_path:
        path_plot = osp.join(artifacts_path, "model.png")
        makepath(path_plot)
        gan.plot(to_file = path_plot)

    return gan

MODELS = [{
    "class": LinearRegression,
    "name": "linear-regression",
    "params": {
        # "verbose": 1
    }
}, {
    "class": GaussianProcessRegressor,
    "name": "gaussian-process-regressor",

}, {
    "class": MLPRegressor,
    "name": "mlp-regressor",
    "params": {
        "hidden_layer_sizes": (100,),
        # "verbose": True
    }
}]

def _train_model_step(model_meta, X_train, X_test, Y_train, Y_test, **kwargs):
    model  = model_meta["class"](**model_meta.get("params", {}))
    k_fold = kwargs.get("k_fold", settings.get("k_fold"))

    logger.info("Training model: %s" % model_meta["name"])

    k_fold = KFold(n_splits = k_fold, shuffle = True)

    for i, (train_index, test_index) in enumerate(k_fold.split(X_train)):
        x_train, x_val = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train, y_val = Y_train.iloc[train_index], Y_train.iloc[test_index]

        logger.info("Training fold %d/%d" % (i + 1, k_fold.n_splits))

        model.fit(x_train, y_train)

        logger.info("Model: %s, Fold: %d, Score: %.4f" % (
            model_meta["name"], i, model.score(x_val, y_val)))

    logger.success("Successfully trained model: %s" % model_meta["name"])

    logger.info("Evaluating model...")

    score = model.score(X_test, Y_test)

    logger.success("Successfully evaluated model: %s with score: %.4f" % (model_meta["name"], score))

def _train_step(csv_path, data_dir = None, objective = False, n_y = None, *args, **kwargs):
    data_dir = get_data_dir(NAME, data_dir)
    jobs = kwargs.get("jobs", settings.get("jobs"))

    logger.info("Training on CSV file: %s" % csv_path)

    test_size = kwargs.get("test_size", settings.get("test_size"))

    df = dfl.read_csv(csv_path)

    logger.success("Loaded CSV file: %s" % csv_path)

    logger.info("Splitting data into train and test sets...")
    train_df, test_df = train_test_split(df, test_size = test_size)

    logger.success("Successfully split data into train and test sets.")

    X_columns = [column for column in df.columns if "flux" not in column]
    
    if objective or n_y:
        model_id = osp.splitext(osp.basename(csv_path))[0]

        cobra_config.cache_directory = data_dir

        path_model = osp.join(data_dir, "%s_minimized.xml" % model_id)

        model_gemm = read_sbml_model(path_model)

        logger.info("Loaded GEMM model: %s" % model_id)

        if objective:
            objectives = linear_reaction_coefficients(model_gemm)
            objective  = list(iterkeys(objectives))[0]

            y_columns  = ["%s_flux" % objective.id]
        else:
            n_reactions = len(model_gemm.reactions)
            reactions   = get_random_model_object_sample(model_gemm, "reactions", n = int(float(n_y) * n_reactions))

            y_columns   = ["%s_flux" % reaction.id for reaction in reactions]
    else:
        y_columns = list(set(df.columns) - set(X_columns))

    X_train, X_test, y_train, y_test = train_df[X_columns], test_df[X_columns], \
        train_df[y_columns], test_df[y_columns]

    logger.info("Starting training...")

    with parallel.no_daemon_pool(processes = jobs) as pool:
        try:
            fn = build_fn(_train_model_step, X_train = X_train, X_test = X_test,
                Y_train = y_train, Y_test = y_test, *args, **kwargs)
            list(pool.map(fn, MODELS))
        except Exception as e:
            logger.error("Failed to train model: %s" % e)
            raise e

def train(check = False, data_dir = None, artifacts_path = None, *args, **kwargs):
    logger.info("Initiating Training...")

    logger.info("Storing artifacts at path: %s" % artifacts_path)

    jobs = kwargs.get("jobs", settings.get("jobs"))

    # batch_size    = 1 if check else settings.get("batch_size")
    # learning_rate = settings.get("learning_rate")
    # epochs        = 1 if check else settings.get("epochs")
    
    # model = build_model(artifacts_path = artifacts_path)
    # model.compile(optimizer = Adam(learning_rate = learning_rate),
    #     loss = dice_loss, metrics = [binary_accuracy])

    # dops.watch(model)

    # output_shape = model.output_shape[1:-1]

    # width, height = settings.get("image_width"), \
    #     settings.get("image_height")

    data_dir  = get_data_dir(NAME, data_dir)

    data_csv  = get_files(data_dir, "*.csv")
    
    if len(data_csv) == 0:
        logger.warn("No CSV file found in directory: %s" % data_dir)
    else:
        logger.info("Found %s CSV files" % len(data_csv))

        with parallel.no_daemon_pool(processes = jobs) as pool:
            try:
                fn = build_fn(_train_step, data_dir = data_dir, *args, **kwargs)
                pool.map(fn, data_csv)
            except Exception as e:
                logger.error("Failed to train model: %s" % e)
                raise e

    # args = dict(
    #     batch_size = batch_size,
    #     # color_mode = "grayscale",
    #     image_size = (width, height),
    #     mask_size  = output_shape,
    #     shuffle    = True
    # )