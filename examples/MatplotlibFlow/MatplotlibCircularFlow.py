import sys
from typing import Dict, Any

from aiflows.base_flows import CircularFlow
from aiflows.utils import logging
from HumanStandardInputFlowModule import HumanStandardInputFlow
from ChatFlowModule import ChatAtomicFlow
from InterpreterFlowModule import InterpreterAtomicFlow
from PopUpPDFFlowModule import PopUpPDFFlow 

logging.set_verbosity_debug()

log = logging.get_logger(__name__)

class MatplotlibCircularFlow(CircularFlow):
    def __init__(self, flow_config: Dict[str, Any], subflows: Dict[str, Any]):
        super().__init__(flow_config=flow_config, subflows=subflows)
        

    def _on_reach_max_round(self):
        """Called when the maximum number of rounds is reached."""
        self._state_update_dict({
            "answer": "Reached the maximum number of iterations without completing the task.",
            "status": "unfinished"
        })

    @staticmethod
    def prepare_plot_details_input(flow_state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare input for collecting plot details."""
        return {"prompt": "Please describe the plot you'd like to create:"}

    @CircularFlow.input_msg_payload_builder
    def prepare_script_generation_input(self, flow_state: Dict[str, Any], dst_flow: ChatAtomicFlow) -> Dict[str, Any]:
        """Prepare input for generating plot script based on user input."""
        plot_details = flow_state.get("plot_details", "")
        return {"plot_details": plot_details}

    @CircularFlow.output_msg_payload_processor
    def process_script_generation_output(self, output_payload: Dict[str, Any], src_flow: ChatAtomicFlow) -> Dict[str, Any]:
        """Process output from script generation to extract the plot script."""
        plot_script = output_payload.get("plot_script", "")
        return {"plot_script": plot_script}

    @CircularFlow.input_msg_payload_builder
    def prepare_script_execution_input(self, flow_state: Dict[str, Any], dst_flow: InterpreterAtomicFlow) -> Dict[str, Any]:
        """Prepare input for executing the plot script."""
        plot_script = flow_state.get("plot_script", "")
        return {"code_to_execute": plot_script}

    @CircularFlow.output_msg_payload_processor
    def process_display_plot_output(self, output_payload: Dict[str, Any], src_flow: PopupPngFlow) -> Dict[str, Any]:
        """Process output after displaying the plot to check if user wants to continue or exit."""
        # Example: Placeholder for output processing logic
        user_decision = output_payload.get("user_decision", "continue")
        if user_decision.lower() == "exit":
            return {"EARLY_EXIT": True}
        return {}


if __name__ == "__main__":
    
    flow_config = {
        "name": "MatplotlibVisualizationCircularFlow",
        "description": "A circular flow for continuously creating and displaying plots based on user input.",
        # Add more configuration as needed
    }
    subflows = {
        "human_input_flow": HumanStandardInputFlow(),
        "chat_flow": ChatAtomicFlow(),
        "interpreter_flow": InterpreterAtomicFlow(),
        "popup_pdf_flow": PopUpPDFFlow(),
    }

    circular_flow = MatplotlibCircularFlow(flow_config=flow_config, subflows=subflows)
    circular_flow.run(input_data={})

