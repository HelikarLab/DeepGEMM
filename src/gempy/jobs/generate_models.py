from bpyutils.util.system import popen
from bpyutils.const import CPU_COUNT
from bpyutils.log import get_logger
from bpyutils import parallel

from gempy.api.refseq import RefSeq
from gempy.__attr__ import __name__ as NAME

REFSEQ = RefSeq(test = False)

logger = get_logger(name = NAME)

def generate_model(refseq_id):
    popen("python -m gempy --verbose --refseq %s" % refseq_id)

def run(*args, **kwargs):
    jobs = kwargs.get("jobs", CPU_COUNT)

    accessions = REFSEQ.accessions

    logger.info("Generating models for %s genomes..." % len(accessions))

    with parallel.no_daemon_pool(processes = jobs) as pool:
        pool.map(generate_model, accessions)