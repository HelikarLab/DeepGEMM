# imports - compatibility imports
from __future__ import absolute_import

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# imports - standard imports

from gempy.commands.util 	    import cli_format
=======
from gempy.commands.util 	import cli_format
>>>>>>> template/master
=======
from gempy.commands.util 	import cli_format
>>>>>>> template/master
=======
from gempy.commands.util 	import cli_format
>>>>>>> template/master
=======
from gempy.commands.util 	import cli_format
>>>>>>> template/master
from bpyutils.util._dict        import merge_dict
from bpyutils.util.system   	import (touch)
from bpyutils.util.error        import pretty_print_error
from bpyutils.config			import environment
from bpyutils.exception         import DependencyNotFoundError
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
from bpyutils import log, parallel
from gempy import cli
from bpyutils._compat		    import iteritems
from gempy.__attr__      	    import __name__
from gempy.commands.helper      import (
    download_refseq,
    process_faa_file
)
=======
=======
>>>>>>> template/master
=======
>>>>>>> template/master
=======
>>>>>>> template/master
from bpyutils import log
from gempy 	import cli
from bpyutils._compat		    import iteritems
from gempy.__attr__ import __name__
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> template/master
=======
>>>>>>> template/master
=======
>>>>>>> template/master
=======
>>>>>>> template/master

logger   = log.get_logger(level = log.DEBUG)

ARGUMENTS = dict(
    jobs						= 1,
    check		 				= False,
    interactive  				= False,
    yes			 				= False,
    no_cache		            = False,
    no_color 	 				= True,
    output						= None,
    ignore_error				= False,
    force						= False,
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    verbose		 				= False,

    faa                         = [],
=======
    verbose		 				= False
>>>>>>> template/master
=======
    verbose		 				= False
>>>>>>> template/master
=======
    verbose		 				= False
>>>>>>> template/master
=======
    verbose		 				= False
>>>>>>> template/master
)

@cli.command
def command(**ARGUMENTS):
    try:
        return _command(**ARGUMENTS)
    except Exception as e:
        if not isinstance(e, DependencyNotFoundError):
            cli.echo()

            pretty_print_error(e)

            cli.echo(cli_format("""\
An error occured while performing the above command. This could be an issue with
"gempy". Kindly post an issue at https://github.com/achillesrasquinha/gempy/issues""", cli.RED))
        else:
            raise e

def to_params(kwargs):
    class O(object):
        pass

    params = O()

    kwargs = merge_dict(ARGUMENTS, kwargs)

    for k, v in iteritems(kwargs):
        setattr(params, k, v)

    return params

def _command(*args, **kwargs):
    a = to_params(kwargs)

    if not a.verbose:
        logger.setLevel(log.NOTSET)

    logger.info("Environment: %s" % environment())
    logger.info("Arguments Passed: %s" % locals())

    file_ = a.output

    if file_:
        logger.info("Writing to output file %s..." % file_)
        touch(file_)
    
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    logger.info("Using %s jobs..." % a.jobs)

    faas = a.faa or []

    if a.refseq:
        logger.info("Found RefSeq accession numbers %s..." % a.refseq)

        with parallel.pool(processes = a.jobs) as pool:
            results = pool.map(download_refseq, a.refseq)
            faas += results

    if faas:
        logger.info("Found %s FAA files..." % len(faas))

        with parallel.pool(processes = a.jobs) as pool:
            pool.map(process_faa_file, faas)
=======
    logger.info("Using %s jobs..." % a.jobs)
>>>>>>> template/master
=======
    logger.info("Using %s jobs..." % a.jobs)
>>>>>>> template/master
=======
    logger.info("Using %s jobs..." % a.jobs)
>>>>>>> template/master
=======
    logger.info("Using %s jobs..." % a.jobs)
>>>>>>> template/master
