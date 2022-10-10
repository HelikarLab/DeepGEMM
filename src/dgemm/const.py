from bpyutils.const import CPU_COUNT
from bpyutils.util.environ import getenv

from dgemm import __name__ as NAME

_PREFIX = NAME.upper()

CONST = {
    "prefix": _PREFIX
}

DEFAULT = {
    "jobs":                 getenv("JOBS", CPU_COUNT, prefix = _PREFIX),
    "batch_size":           256,
    "learning_rate":        1e-4,
    "batch_norm":           True,
    "epochs":               50,

    "encoder_dropout_rate": 0.3,
    "encoder_batch_norm":   False,
    "decoder_dropout_rate": 0,
    "decoder_batch_norm":   True,

    "min_react_reaction_retention_list": ("ATPM",),

    "test_size":            0.2,
    "k_fold":               5
}