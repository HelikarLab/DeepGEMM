# imports - compatibility imports
from __future__ import absolute_import

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# imports - module imports
from bpyutils import log

from gempy.api.refseq import RefSeq
from gempy.api.blast  import BLAST

logger  = log.get_logger()

def download_refseq(id_):
    refseq   = RefSeq()
    filepath = refseq.download(id_)
    return filepath

def process_faa_file(faa):
    blast   = BLAST()
    results = blast.blastp(faa)
=======
=======
>>>>>>> template/master
=======
>>>>>>> template/master
=======
>>>>>>> template/master
# imports - standard imports

# imports - module imports
from bpyutils import log

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
logger = log.get_logger()
>>>>>>> template/master
=======
logger = log.get_logger()
>>>>>>> template/master
=======
logger = log.get_logger()
>>>>>>> template/master
=======
logger = log.get_logger()
>>>>>>> template/master
