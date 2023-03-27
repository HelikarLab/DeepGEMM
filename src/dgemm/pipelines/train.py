import os.path as osp
from statistics import LinearRegression
import warnings
import random

from upyog.util.system import makepath, get_files
from upyog.util.types import build_fn
from upyog.util.ml import get_data_dir
from upyog.log import get_logger
from upyog.const import CPU_COUNT
from upyog._compat import iterkeys
from upyog import parallel

from cobra.io.web import load_model as load_gemm
from cobra.io import read_sbml_model
from cobra.util import linear_reaction_coefficients
import cobra

# import deeply

from dgemm import settings, __name__ as NAME
from dgemm import settings # , dops

warnings.filterwarnings("ignore")

logger = get_logger(NAME)

cobra_config = cobra.Configuration()

from deeply.integrations.imports import import_ds_module

dfl = import_ds_module("pandas")
KFold = import_ds_module("sklearn.model_selection.KFold")
train_test_split = import_ds_module("sklearn.model_selection.train_test_split")

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
    
    if artifacts_path:
        path_plot = osp.join(artifacts_path, "model.png")
        makepath(path_plot)
        gan.plot(to_file = path_plot)

    return gan

MODELS = [{
    "class": import_ds_module("sklearn.linear_model.LinearRegression"),
    "name": "linear-regression"
}, {
    "class": import_ds_module("sklearn.gaussian_process.GaussianProcessRegressor"),
    "name": "gaussian-process-regressor"
}, {
    "class": import_ds_module("sklearn.ensemble.RandomForestRegressor"),
    "name": "random-forest-regressor" 
}, {
    "class": import_ds_module("sklearn.svm.SVR"),
    "name": "support-vector-regressor" 
}, {
    "class": import_ds_module("sklearn.neural_network.MLPRegressor"),
    "name": "mlp-regressor",
    "params": {
        "hidden_layer_sizes": (100,),
        "verbose": True
    }
}]

def _train_model_step(model_meta, X_train, X_test, Y_train, Y_test, **kwargs):
    model  = model_meta["class"](**model_meta.get("params", {}))
    k_fold = kwargs.get("k_fold", settings.get("k_fold"))

    logger.info("Watching model: %s" % model_meta["name"])
    # dops.watch(model)

    logger.info("Training model: %s" % model_meta["name"])

    k_fold = KFold(n_splits = k_fold, shuffle = True)

    for i, (train_index, test_index) in enumerate(k_fold.split(X_train)):
        x_train, x_val = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train, y_val = Y_train.iloc[train_index], Y_train.iloc[test_index]

        logger.info("Training fold %d/%d" % (i + 1, k_fold.n_splits))

        model.fit(x_train, y_train)

        logger.info("Model: %s, Fold: %d, Score: %.4f" % (
            model_meta["name"], i, model.score(x_val, y_val) * 100))

    logger.success("Successfully trained model: %s" % model_meta["name"])

    logger.info("Evaluating model...")

    score = model.score(X_test, Y_test)

    logger.success("Successfully evaluated model: %s with score: %.4f" % (model_meta["name"], score * 100))

def _train_step(csv_path, data_dir = None, objective = False, n_y = None, *args, **kwargs):
    data_dir  = get_data_dir(NAME, data_dir)
    jobs      = kwargs.get("jobs", settings.get("jobs"))
    minimized = kwargs.get("minimized", False)

    logger.info("Training on CSV file: %s" % csv_path)

    test_size = kwargs.get("test_size", settings.get("test_size"))

    df = dfl.read_csv(csv_path)

    logger.success("Loaded CSV file: %s" % csv_path)

    logger.info("Splitting data into train and test sets...")
    train_df, test_df = train_test_split(df, test_size = test_size)

    logger.success("Successfully split data into train and test sets.")

    X_columns = [column for column in df.columns if "flux" not in column]
    y_columns = []
    
    if objective or n_y:
        model_id = osp.splitext(osp.basename(csv_path))[0]

        cobra_config.cache_directory = data_dir

        if minimized:
            path_model = osp.join(data_dir, "%s_minimized.xml" % model_id)
            model_gemm = read_sbml_model(path_model)
        else:
            model_gemm = load_gemm(model_id)

        logger.info("Loaded GEMM model: %s" % model_id)

        if n_y:
            y_columns  = list(set(df.columns) - set(X_columns))
            y_columns  = random.sample(y_columns, int(len(y_columns) * n_y))

        if objective:
            objectives = linear_reaction_coefficients(model_gemm)
            objective  = list(iterkeys(objectives))[0]

            y_columns += ["%s_flux" % objective.id]
    else:
        y_columns = list(set(df.columns) - set(X_columns))
            
    logger.info("Using n(y) columns: %s" % len(y_columns))

    X_train, X_test, y_train, y_test = train_df[X_columns], test_df[X_columns], \
        train_df[y_columns], test_df[y_columns]

    logger.info("Starting training...")

    with parallel.no_daemon_pool(processes = jobs) as pool:
        fn = build_fn(_train_model_step, X_train = X_train, X_test = X_test,
            Y_train = y_train, Y_test = y_test, *args, **kwargs)
        list(pool.map(fn, MODELS))

def train(data_dir = None, artifacts_path = None, *args, **kwargs):
    logger.info("Initiating Training...")

    logger.info("Storing artifacts at path: %s" % artifacts_path)

    jobs = kwargs.get("jobs", settings.get("jobs"))

    data_dir  = get_data_dir(NAME, data_dir)

    data_csv  = get_files(data_dir, "*.csv")
    
    if len(data_csv) == 0:
        logger.warn("No CSV file found in directory: %s" % data_dir)
    else:
        logger.info("Found %s CSV files" % len(data_csv))

        with parallel.no_daemon_pool(processes = jobs) as pool:
            fn = build_fn(_train_step, data_dir = data_dir, *args, **kwargs)
            list(pool.map(fn, data_csv))