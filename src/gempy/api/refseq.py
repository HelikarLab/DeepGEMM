import os.path as osp

from bpyutils.api.base      import BaseAPI
from bpyutils.util.types    import lmap
from bpyutils.util.request  import download_file
from bpyutils.util._csv     import read as read_csv
from bpyutils.util.system   import (
    read as read_file,
    write as write_file,
    get_basename
)
from bpyutils.util.array    import find
from bpyutils.log           import get_logger
from bpyutils._compat       import urljoin
from bpyutils.db            import get_connection
from dgemm.__attr__         import __name__ as NAME
from dgemm.config           import PATH

logger = get_logger(name = NAME)

PATH_CACHE = osp.join(PATH["CACHE"], "refseq")
db = get_connection(location = PATH["CACHE"])

class RefSeq(BaseAPI):
    url = "https://ftp.ncbi.nlm.nih.gov/genomes/refseq"

    def __init__(self, *args, **kwargs):
        kwargs["test"] = kwargs.get("test", False)

        self._super = super(RefSeq, self)
        self._super.__init__(*args, **kwargs)

    @property
    def list(self):
        table   = db["RefSeqSummary"]
        results = table.all()

        if not results:
            path = osp.join(PATH_CACHE, "refseq.tsv")
            
            if not osp.exists(path):
                url = self._build_url("assembly_summary_refseq.txt")

                logger.info("Downloading RefSeq summary...")
                download_file(url, path = path)

                logger.info("Sanitizing RefSeq summary...")
                content = read_file(path)
                lines   = content.splitlines()
                
                # sanitize
                lines    = lines[1:]
                lines[0] = lines[0][2:]

                write_file(path, "\n".join(lines), force = True)

            logger.info("Loading accession list...")
            data = read_csv(path, delimiter = "\t")

            table.insert(data)

            results = table.all()

        return results

    @property
    def accessions(self):
        return lmap(lambda x: x["assembly_accession"], self.list)

    def download(self, id_):
        metadata = find(self.list, lambda x: x["assembly_accession"] == id_)

        prefix = "protein.faa.gz"
        path_target = osp.join(PATH_CACHE, id_, prefix)

        if metadata:
            if not osp.exists(path_target):
                ftp_path = metadata["ftp_path"]

                logger.info("Downloading RefSeq %s to %s" % (id_, path_target))
                base_name = get_basename(ftp_path)

                url = self._build_url(ftp_path, "%s_%s" % (base_name, prefix), prefix = False)
                download_file(url, path = path_target)
            else:
                logger.warn("%s already available." % path_target)
        else:
            raise ValueError("No accession number %s found." % id_)

        return path_target