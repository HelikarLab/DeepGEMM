from upyog.util.system import popen
from upyog.const import CPU_COUNT
from upyog.log import get_logger
from upyog import parallel

from dgemm.api.refseq import RefSeq
from dgemm.__attr__ import __name__ as NAME

REFSEQ = RefSeq(test = False)

logger = get_logger(name = NAME)

def generate_model(refseq_id):
    popen("python -m dgemm --verbose --refseq %s" % refseq_id)

def run(*args, **kwargs):
    jobs = kwargs.get("jobs", CPU_COUNT)

    accessions = REFSEQ.accessions

    logger.info("Generating models for %s genomes..." % len(accessions))

    with parallel.no_daemon_pool(processes = jobs) as pool:
        pool.map(generate_model, accessions)