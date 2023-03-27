from __future__ import absolute_import

try:
    import os

    if os.environ.get("DGEMM_GEVENT_PATCH"):
        from gevent import monkey
        monkey.patch_all(threaded = False, select = False)
except ImportError:
    pass

# imports - module imports
from dgemm.__attr__ import (
    __name__,
    __version__,
    __build__,

    __description__,

    __author__
)
from dgemm.config      import PATH
from dgemm.const       import DEFAULT
from dgemm.__main__    import main

from upyog.cache       import Cache
from upyog.config      import Settings
from upyog.util.jobs   import run_all as run_all_jobs, run_job
from upyog import log

cache = Cache(dirname = __name__)
cache.create()

logger   = log.get_logger(__name__)

settings = Settings(location = PATH["CACHE"], defaults = {
    "jobs":                 DEFAULT["jobs"],
    "batch_size":           DEFAULT["batch_size"],
    "learning_rate":        DEFAULT["learning_rate"],
    "encoder_dropout_rate": DEFAULT["encoder_dropout_rate"],
    "encoder_batch_norm":   DEFAULT["encoder_batch_norm"],
    "decoder_dropout_rate": DEFAULT["decoder_dropout_rate"],
    "decoder_batch_norm":   DEFAULT["decoder_batch_norm"],
    "epochs":               DEFAULT["epochs"],

    "test_size":            DEFAULT["test_size"],
    "k_fold":               DEFAULT["k_fold"]
})

def get_version_str():
    version = "%s%s" % (__version__, " (%s)" % __build__ if __build__ else "")
    return version

if os.environ.get("DGEMM_WANDB"):
    import deeply
    dops = deeply.ops.service("wandb")
    dops.init("dgemm")