# imports - compatibility imports
from __future__ import absolute_import

# imports - module imports
from upyog import log

from dgemm.api.refseq import RefSeq
from dgemm.api.blast  import BLAST

logger  = log.get_logger()

def download_refseq(id_):
    refseq   = RefSeq()
    filepath = refseq.download(id_)
    return filepath

def process_faa_file(faa):
    blast   = BLAST()
    results = blast.blastp(faa)
