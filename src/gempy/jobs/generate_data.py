import os.path as osp

from bpyutils.util.system  import (
    make_temp_dir,
    extract_all,
    makedirs
)
from bpyutils.util.environ import getenv
from bpyutils.util.git import update_git_repo
from bpyutils.config   import get_config_path
from bpyutils.log      import get_logger

from gempy.__attr__ import __name__ as NAME

logger = get_logger(name = NAME)

def run(*args, **kwargs):
    logger.info("Generating data...")

    PATH_CACHE  = get_config_path(NAME)

    dir_path    = PATH_CACHE

    repo = osp.join(dir_path, "gempy")

    github_username    = getenv("JOBS_GITHUB_USERNAME",    prefix = NAME.upper(), raise_err = True)
    github_oauth_token = getenv("JOBS_GITHUB_OAUTH_TOKEN", prefix = NAME.upper(), raise_err = True)

    url  = "https://github.com/achillesrasquinha/gempy"

    update_git_repo(repo, clone = True, url = url,
        username = github_username, password = github_oauth_token,
        branch = "develop")

    path_data_base = osp.join(repo, "data")
    path_data = osp.join(path_data_base, "data.tar.gz")

    with make_temp_dir() as tmp_dir:
        data_dir = osp.join(tmp_dir, "data")
        makedirs(data_dir)

        if osp.exists(path_data):
            extract_all(path_data, data_dir)