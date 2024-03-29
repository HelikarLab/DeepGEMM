import os, os.path as osp
import threading

from upyog.util.system  import (
    make_temp_dir,
    extract_all,
    makedirs,
    list_tree,
    copy as copy_files,
    make_archive,
    pardir
)
from upyog.util.datetime import get_timestamp_str
from upyog.util.environ  import getenv
from upyog.util.git import update_git_repo, commit
from upyog.config   import get_config_path
from upyog.log      import get_logger
from upyog.const    import CPU_COUNT

from dgemm.data.functions import (
    fetch_models,
    generate_flux_data
)

from dgemm.__attr__ import __name__ as NAME

logger = get_logger(name = NAME)

DEFAULT_BRANCH  = "develop"
COMMIT_INTERVAL = 60 * 5

def save_and_publish(repo_dir, data_dir, target_path):
    timer = threading.Timer(COMMIT_INTERVAL, save_and_publish, [repo_dir, data_dir, target_path])
    timer.start()
    
    with make_temp_dir() as tmp_dir:
        if osp.exists(data_dir):
            logger.info("Creating an archive...")

            copy_files(data_dir, dest = tmp_dir, recursive = True)
            make_archive(target_path, "gztar", tmp_dir)

            commit(repo_dir,
                message     = "[skip ci]: Update database - %s" % get_timestamp_str(),
                allow_empty = True,
                add = target_path, push = True, branch = DEFAULT_BRANCH
            )
        else:
            logger.warn("Couldn't find data directory: %s" % data_dir)

def run(*args, **kwargs):
    logger.info("Generating data...")

    jobs  = kwargs.get("jobs",  CPU_COUNT)
    check = kwargs.get("check", False)
    gen_flux_data = kwargs.get("gen_flux_data", False)

    PATH_CACHE = get_config_path(NAME)

    dir_path   = PATH_CACHE

    repo = osp.join(dir_path, "dgemm")

    github_username    = getenv("JOBS_GITHUB_USERNAME",    prefix = NAME.upper(), raise_err = True)
    github_oauth_token = getenv("JOBS_GITHUB_OAUTH_TOKEN", prefix = NAME.upper(), raise_err = True)

    url  = "https://github.com/achillesrasquinha/dgemm"

    logger.info("Updating repo...")

    update_git_repo(repo, clone = True, url = url,
        username = github_username, password = github_oauth_token,
        branch = DEFAULT_BRANCH)

    path_data_base = osp.join(repo, "data")
    path_data = osp.join(path_data_base, "data.tar.gz")

    with make_temp_dir() as tmp_dir:
        data_dir    = osp.join(tmp_dir,  "data")
        models_dir  = osp.join(data_dir, "models")

        makedirs(models_dir)

        if osp.exists(path_data):
            logger.info("Extracting previous data...")
            extract_all(path_data, tmp_dir)

        logger.info("Beginning save_and_publsih thread...")

        save_and_publish_thread = threading.Thread(target = save_and_publish,
            name = "save_and_publish", kwargs = { "repo_dir": repo,
                "data_dir": tmp_dir, "target_path": path_data })
        save_and_publish_thread.start()

        fetch_models(data_dir = models_dir, check = check, gen_flux_data = gen_flux_data,
            flux_data_dir = data_dir, jobs = jobs)