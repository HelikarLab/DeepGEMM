import os.path as osp

from bpyutils.model.base import BaseObject
from bpyutils.config import get_config_path
from bpyutils.util.system import makedirs, popen

from gempy.__attr__ import name as NAME
from gempy.config import DEFAULT

def _download_blast_db(name, path):
    popen("update_blastdb.pl --decompress --blastdb_version 5 %s" % name,
        cwd = path)

class Backend(BaseObject):
    def blastp(self):
        raise NotImplementedError

class Diamond(Backend):
    def __init__(self, *args, **kwargs):
        self._database = kwargs.get("database", DEFAULT["diamond_db"])
    
    def _check_db(self):
        path_config = get_config_path(NAME)
        path_config_diamond_db = osp.join(path_config, "diamond", "db", self._database)

        if not osp.exists(path_config_diamond_db):
            makedirs(path_config_diamond_db)
            _download_blast_db(self._database, path_config_diamond_db)

    def blastp(self, input_):
        self._check_db()

BACKENDS = {
    "diamond": {
        "class": Diamond
    }
}

class BLAST(BaseObject):
    def __init__(self, *args, **kwargs):
        backend = kwargs.pop("backend", "diamond")

        if backend not in BACKENDS:
            raise ValueError("Backend %s not found." % backend)

        self._super = super(BLAST, self)
        self._super.__init__(*args, **kwargs)

        backend_meta = BACKENDS[backend]
        self.backend = backend_meta["class"]()

    def blastp(self, input_):
        self.backend.blastp(input_)