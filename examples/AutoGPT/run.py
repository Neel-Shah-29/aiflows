import os

import hydra

import aiflows
from aiflows.flow_launchers import FlowLauncher
from aiflows.backends.api_info import ApiInfo
from aiflows.utils.general_helpers import read_yaml_file

from aiflows import logging
from aiflows.flow_cache import CACHING_PARAMETERS, clear_cache

CACHING_PARAMETERS.do_caching = False  # Set to True in order to disable caching
# clear_cache() # Uncomment this line to clear the cache

logging.set_verbosity_debug()

from aiflows import flow_verse
# ~~~ Load Flow dependecies from FlowVerse ~~~
dependencies = [
    {"url": "aiflows/AutoGPTFlowModule", "revision": "f56bea985728b3b12d1042873abadfa9ebd4b4f6"},
    {"url": "aiflows/LCToolFlowModule", "revision": "f1020b23fe2a1ab6157c3faaf5b91b5cdaf02c1b"},
]

flow_verse.sync_dependencies(dependencies)
if __name__ == "__main__":
    # ~~~ Set the API information ~~~
    # OpenAI backend
    api_information = [ApiInfo(backend_used="openai", api_key=os.getenv("OPENAI_API_KEY"))]
    # Azure backend
    # api_information = ApiInfo(backend_used = "azure",
    #                           api_base = os.getenv("AZURE_API_BASE"),
    #                           api_key = os.getenv("AZURE_OPENAI_KEY"),
    #                           api_version =  os.getenv("AZURE_API_VERSION") )

    root_dir = "."
    cfg_path = os.path.join(root_dir, "AutoGPT.yaml")
    cfg = read_yaml_file(cfg_path)
    
    # put the API information in the config
    cfg["flow"]["subflows_config"]["Controller"]["backend"]["api_infos"] = api_information
    cfg["flow"]["subflows_config"]["Memory"]["backend"]["api_infos"] = api_information
    # ~~~ Instantiate the Flow ~~~
    flow_with_interfaces = {
        "flow": hydra.utils.instantiate(cfg["flow"], _recursive_=False, _convert_="partial"),
        "input_interface": (
            None
            if cfg.get("input_interface", None) is None
            else hydra.utils.instantiate(cfg["input_interface"], _recursive_=False)
        ),
        "output_interface": (
            None
            if cfg.get("output_interface", None) is None
            else hydra.utils.instantiate(cfg["output_interface"], _recursive_=False)
        ),
    }

    # ~~~ Get the data ~~~
    # data = {"id": 0, "goal": "Answer the following question: What is the population of Canada?"}  # Uses wikipedia
    # data = {"id": 0, "goal": "Answer the following question: Who was the NBA champion in 2023?"}  # Uses duckduckgo
    data = {
        "id": 0,
        "goal": "Answer the following question: What is the profession and date of birth of Michael Jordan?",
    }
    # At first, we retrieve information about Michael Jordan the basketball player
    # If we provide feedback, only in the first round, that we are not interested in the basketball player,
    #   but the statistician, and skip the feedback in the next rounds, we get the correct answer

    # ~~~ Run inference ~~~
    path_to_output_file = None
    # path_to_output_file = "output.jsonl"  # Uncomment this line to save the output to disk

    _, outputs = FlowLauncher.launch(
        flow_with_interfaces=flow_with_interfaces,
        data=data,
        path_to_output_file=path_to_output_file,
    )

    # ~~~ Print the output ~~~
    flow_output_data = outputs[0]
    print(flow_output_data)
