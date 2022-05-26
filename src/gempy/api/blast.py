from bpyutils.model.base import BaseObject

class Backend(BaseObject):
    def blastp(self):
        raise NotImplementedError

class Diamond(Backend):
    def __init__(self, *args, **kwargs):
        pass

    def blastp(self):

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