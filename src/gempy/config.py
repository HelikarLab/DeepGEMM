from __future__ import absolute_import

import os.path as osp

from gempy.__attr__ import __name__ as NAME

from bpyutils.config      import get_config_path
from bpyutils.util.system import pardir

PATH = dict()

PATH["BASE"]  = pardir(__file__, 1)
PATH["DATA"]  = osp.join(PATH["BASE"], "data")
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
PATH["CACHE"] = get_config_path(NAME)

DEFAULT = {
    "diamond_db": "pdbaa",
    "bigg_model_id": "e_coli_core"
}
=======
PATH["CACHE"] = get_config_path(NAME)
>>>>>>> template/master
=======
PATH["CACHE"] = get_config_path(NAME)
>>>>>>> template/master
=======
PATH["CACHE"] = get_config_path(NAME)
>>>>>>> template/master
