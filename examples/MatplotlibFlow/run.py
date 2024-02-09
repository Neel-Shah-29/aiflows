import os
import hydra

from aiflows import logging
from aiflows.utils.general_helpers import read_yaml_file
from aiflows.flow_launchers import FlowLauncher


from MatplotlibCircularFlow import MatplotlibCircularFlow

logging.set_verbosity_debug()

if __name__ == "__main__":
    root_dir = "."
    cfg_path = os.path.join(root_dir, "MatplotlibCircularFlow.yaml")
    cfg = read_yaml_file(cfg_path)

    # Instantiate the MatplotlibCircularFlow with the loaded configuration
    matplotlib_circular_flow = hydra.utils.instantiate(cfg['flow'], _recursive_=False, _convert_="partial")

    # Define input data for the flow; in this case, it might be initial parameters for the plot
    data = {
        "id": 0,
        "dataset_csv": "path/to/your/dataset.csv",
        "visualization_instructions": "Please describe the type of plot you would like to generate."
    }

    # Optionally, specify a path to output file if you wish to save the output
    path_to_output_file = None  

    _, outputs = FlowLauncher.launch(
        flow_with_interfaces={"flow": matplotlib_circular_flow}, 
        data=data, 
        path_to_output_file=path_to_output_file
    )
