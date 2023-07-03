import os
import re
import shutil
import logging
import inspect

import colorama

logger = logging.getLogger(__name__)

default_home = os.path.join(os.path.expanduser("~"), ".cache")
flows_cache_home = os.path.expanduser(os.path.join(default_home, "flows"))
DEFAULT_CACHE_PATH = os.path.join(flows_cache_home, "flow_verse")
DEFAULT_FLOW_MODULE_FOLDER = "flow_modules"
MODULE_ID_FILE_NAME = "FLOW_MODULE_ID"
REVISION_FILE_HEADER="""\
########################################
# auto-generated by flows, DO NOT EDIT #
########################################
"""
LOCAL_REVISION="local"
DEFAULT_REMOTE_REVISION="main"

import sys
import importlib
import huggingface_hub

from typing import List, Dict

# TODO(yeeef): delete legacy functions
def add_to_sys_path(path):
    # Make sure the path is absolute
    absolute_path = os.path.abspath(path)

    # Check if the path is in sys.path
    if absolute_path not in sys.path:
        # If it's not, add it
        sys.path.append(absolute_path)


def _is_local_path(path_to_dir):
    """Returns True if path_to_dir is a path to a local directory."""

    # check if the directory exists
    if os.path.isdir(path_to_dir):
        # check if directory is not empty
        if os.listdir(path_to_dir):
            return True
        else:
            return False
    else:
        return False


def _sync_repository(repository_id, cache_dir=DEFAULT_CACHE_PATH, local_dir=None, override=False, **kwargs):
    if override:
        path_to_local_repository = huggingface_hub.snapshot_download(repository_id, cache_dir=cache_dir, local_dir=local_dir,  **kwargs)
    elif _is_local_path(repository_id):
        path_to_local_repository = repository_id
    else:
        path_to_local_repository = huggingface_hub.snapshot_download(repository_id, cache_dir=cache_dir, local_dir=local_dir,  **kwargs)

    logger.warn(f"The flow repository was synced to:{path_to_local_repository}", )  # ToDo: Replace with log.info once the logger is set up
    return path_to_local_repository


def load_config(repository_id, class_name, cache_dir=DEFAULT_CACHE_PATH, local_dir=None, overrides={}):
    flow_class = load_class(repository_id=repository_id,
                            class_name=class_name,
                            local_dir=local_dir,
                            cache_dir=cache_dir)

    config = flow_class.get_config(**overrides)

    return config


def load_class(repository_id, class_name, cache_dir=DEFAULT_CACHE_PATH, local_dir=None):
    logger.warn(f"{colorama.Fore.RED}[load_class] is deprecated, please use [sync_dependencies] and normal import{colorama.Style.RESET_ALL}")
    path_to_local_repository = _sync_repository(repository_id,
                                                local_dir=local_dir,
                                                cache_dir=cache_dir)

    # split local_repo_path into parent and folder name
    local_repo_path_parent = os.path.dirname(path_to_local_repository)
    local_repo_dir_name = os.path.basename(path_to_local_repository)

    add_to_sys_path(local_repo_path_parent)
    flow_module = importlib.import_module(local_repo_dir_name)
    flow_class = getattr(flow_module, class_name)

    return flow_class


def instantiate_flow(repository_id, class_name, cache_dir=DEFAULT_CACHE_PATH, local_dir=None, overrides={}): # TODO(yeeef): this default value might cause problems
    logger.warn(f"{colorama.Fore.RED}[initiate_flow] is deprecated, please use [sync_dependencies] and normal import{colorama.Style.RESET_ALL}")
    flow_class = load_class(repository_id=repository_id,
                            local_dir=local_dir,
                            class_name=class_name,
                            cache_dir=cache_dir)

    config = flow_class.get_config(**overrides)

    return flow_class.instantiate_from_config(config)

def validate_and_augment_dependency(dependency: Dict[str, str]):
    if "url" not in dependency:
        raise ValueError("dependency must have a `url` field")

    if os.path.exists(dependency["url"]):
        dependency["revision"] = LOCAL_REVISION

    if "revision" not in dependency:
        dependency["revision"] = DEFAULT_REMOTE_REVISION


    return dependency

def write_or_append_gitignore(local_dir: str, mode: str, content: str):
    gitignore_path = os.path.join(local_dir, ".gitignore")
    with open(gitignore_path, mode) as gitignore_f:
        lines = [
            "\n\n\n# auto-generated by flows, all synced moduels will be ignored by default\n",
            f"{content}\n"
        ]
        gitignore_f.writelines(lines)

def write_mod_id(local_dir: str, mod_id: str):
    revision_file_path = os.path.join(local_dir, MODULE_ID_FILE_NAME)
    with open(revision_file_path, "w") as revision_f:
        lines = [
            REVISION_FILE_HEADER,
            mod_id,
        ]
        revision_f.writelines(lines)
        revision_f.write("\n")

def read_mod_id(local_dir: str):
    revision_file_path = os.path.join(local_dir, MODULE_ID_FILE_NAME)
    if not os.path.exists(revision_file_path):
        return None
    
    with open(revision_file_path, "r") as revision_f:
        lines = revision_f.readlines()
        if len(lines) != 4:
            return None
        
        if "".join(lines[:3]) != REVISION_FILE_HEADER:
            return None
        
        return lines[3].strip()

def fetch_remote(repo_id: str, revision: str, flow_mod_id: str, cache_dir: str, local_dir: str):
    huggingface_hub.snapshot_download(repo_id, cache_dir=cache_dir, local_dir=local_dir, revision=revision)
    # write the revision info in the folder
    write_mod_id(local_dir, flow_mod_id)
    # add FLOW_MODULE_ID to .gitignore
    write_or_append_gitignore(local_dir, "a", MODULE_ID_FILE_NAME)

def fetch_local(file_path: str, revision: str, flow_mod_id: str, cache_dir: str, local_dir: str, overwrite: bool=False):
    shutil.copytree(file_path, local_dir, ignore=shutil.ignore_patterns(".git"), dirs_exist_ok=overwrite)
    # write the revision info in the folder
    write_mod_id(local_dir, flow_mod_id)

def build_mod_id(repo_id_or_file_path: str, revision: str):
    match = re.search(r"\W", revision)
    if match is not None:
        raise ValueError(f"revision {revision} is not valid, it contains illegal characters: {match.group(0)}")
    
    return f"{repo_id_or_file_path}:{revision}"

def sync_dependency(dep: Dict[str, str], overwrite: bool=False):
    url, revision = dep["url"], dep["revision"]

    mod_id = build_mod_id(url, revision)

    if overwrite:
        logger.warn(f"{colorama.Fore.RED}{mod_id} will be overwritten, are you sure? (Y/N){colorama.Style.RESET_ALL}")
        user_input = input()
        if user_input != "Y":
            overwrite = False

    if os.path.exists(url): # url is a local path
        file_path = url
        if revision != LOCAL_REVISION:
            raise ValueError(f"revision {revision} is not valid for local dependency {file_path}, only {LOCAL_REVISION} is allowed")

        basename = os.path.join(LOCAL_REVISION, file_path.split("/")[-1])

        if not os.path.isdir(file_path):
            raise ValueError(f"dependency url {file_path} is not a valid local directory or huggingface hub repository")
        
        
        local_dir = os.path.join(os.path.curdir, DEFAULT_FLOW_MODULE_FOLDER, basename)
        if not os.path.isdir(local_dir) or overwrite:
            fetch_local(file_path, revision, mod_id, DEFAULT_CACHE_PATH, local_dir, overwrite)
        else:
            logger.warn(f"{mod_id} already synced, skip")


    else: # huggingface url
        repo_id = url
        basename = repo_id
        # check revision
        # check if basename exists in local dir
        local_dir = os.path.join(os.path.curdir, DEFAULT_FLOW_MODULE_FOLDER, basename)
        mod_id = build_mod_id(url, revision)

        if not os.path.isdir(local_dir) or overwrite:
            # first fetch / overwrite
            logger.warn(f"first fetch / overwrite {mod_id}")
            fetch_remote(repo_id, revision, mod_id, DEFAULT_CACHE_PATH, local_dir)
            
        elif os.path.isdir(local_dir) and read_mod_id(local_dir) != mod_id:
            # local dir exists, but revision is not the same, overwrite with new revision
            logger.warn(f"{colorama.Fore.RED}{read_mod_id(local_dir)} already synced, it will be overwritten by new revision {mod_id}, are you sure? (Y/N){colorama.Style.RESET_ALL}")
            user_input = input()
            if user_input == "Y":
                fetch_remote(repo_id, revision, mod_id, DEFAULT_CACHE_PATH, local_dir)
        else:
            logger.warn(f"{mod_id} already synced, skip") 

def sync_dependencies(dependencies: List[Dict[str, str]], all_overwrite: bool=False):
    frame = inspect.currentframe().f_back
    module_name = inspect.getmodule(frame).__name__

    logger.warn(f"{colorama.Fore.GREEN}[{module_name}]{colorama.Style.RESET_ALL} started to sync flow module dependencies...")
    flow_module_dir = os.path.join(os.path.curdir, DEFAULT_FLOW_MODULE_FOLDER)
    if not os.path.exists(flow_module_dir):
        os.mkdir(flow_module_dir)
    elif not os.path.isdir(flow_module_dir):
        raise ValueError(f"flow module folder {flow_module_dir} is not a directory")
    
    write_or_append_gitignore(flow_module_dir, "w", content="*")

    for dep in dependencies:
        dep = validate_and_augment_dependency(dep)
        dep_overwrite = dep.get("overwrite", False)
        sync_dependency(dep, all_overwrite or dep_overwrite)

    logger.warn(f"{colorama.Fore.GREEN}[{module_name}]{colorama.Style.RESET_ALL} finished syncing\n\n")


