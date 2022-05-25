from bpyutils.model.base import BaseObject

class BLAST(BaseObject):
    def __init__(self, *args, **kwargs):
        backend = kwargs.get("backend", "diamond")

        self._super = super(BLAST, self)
        self._super.__init__(*args, **kwargs)

    def blastp(self):
        pass