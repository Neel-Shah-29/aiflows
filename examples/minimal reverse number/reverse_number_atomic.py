import os
from typing import Dict, Any

import flows
from flows.flow_launchers import FlowLauncher
from flows.base_flows import AtomicFlow

from flows.utils.general_helpers import read_yaml_file


# logging.set_verbosity_debug()  # Uncomment this line to see verbose logs


class ReverseNumberAtomicFlow(AtomicFlow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:

        input_number = input_data["number"]
        output_number = int(str(input_number)[::-1])
        response = {"output_number": output_number}
        return response


if __name__ == "__main__":
    root_dir = "."
    cfg_path = os.path.join(root_dir, "reverseNumberAtomic.yaml")
    cfg = read_yaml_file(cfg_path)

    # ~~~ Instantiate the flow ~~~
    flow = ReverseNumberAtomicFlow.instantiate_from_default_config(**cfg)

    # ~~~ Get the data ~~~
    data = {"id": 0, "number": 1234}  # This can be a list of samples

    # ~~~ Run inference ~~~
    path_to_output_file = None
    # path_to_output_file = "output.jsonl"
    _, outputs = FlowLauncher.launch(
        flow_with_interfaces={"flow": flow}, data=data, path_to_output_file=path_to_output_file
    )

    # ~~~ Print the output ~~~
    flow_output_data = outputs[0]
    print(flow_output_data)
