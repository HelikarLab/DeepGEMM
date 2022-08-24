from bpyutils.const import CPU_COUNT
from bpyutils.util.environ import getenv

from gempy import __name__ as NAME

_PREFIX = NAME.upper()

CONST = {
    "prefix": _PREFIX
}

DEFAULT = {
    "jobs":                 getenv("JOBS", CPU_COUNT, prefix = _PREFIX),
    "batch_size":           256,
    "learning_rate":        1e-4,
    "batch_norm":           True,
    "dropout_rate":         0.3,
    "epochs":               50
}