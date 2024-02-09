from copy import deepcopy
import hydra

import time

from typing import Dict, Optional, Any

from aiflows.base_flows import AtomicFlow

from aiflows.utils import logging
from aiflows.messages.flow_message import UpdateMessage_ChatMessage

from aiflows.prompt_template import JinjaPrompt

from aiflows.backends.llm_lite import LiteLLMBackend

log = logging.get_logger(__name__)


class ChatAtomicFlow(AtomicFlow):
    """ This class implements a ChatAtomicFlow. It is a flow that uses a LLM via an API to generate textuals responses to textual inputs.
    It employs litellm as a backend to query the LLM via an API. See litellm's supported models and APIs here: https://docs.litellm.ai/docs/providers
    
    *Configuration Parameters*:
    
    - `name` (str): The name of the flow. This name is used to identify the flow in the logs. Default: "ChatAtomicFlow"
    - `description` (str): A description of the flow. This description is used to generate the help message of the flow. 
    Default: "Flow which uses as tool an LLM though an API"
    - `enable_cache` (bool): Whether to enable cache for this flow. Default: True
    - `n_api_retries` (int): The number of times to retry the API call in case of failure. Default: 6
    - `wait_time_between_retries` (int): The number of seconds to wait between each API call when the call fails. Default: 20
    - `system_name` (str): The name of the system (roles of LLM). Default: "system"
    - `user_name` (str): The name of the user (roles of LLM). Default: "user"
    - `assistant_name` (str): The name of the assistant (roles of LLM). Default: "assistant"
    - `backend` Dict[str,Any]: The backend of the flow. Used to call models via an API.
    See litellm's supported models and APIs here: https://docs.litellm.ai/docs/providers.
    The default parameters of the backend are all defined at aiflows.backends.llm_lite.LiteLLMBackend
    (also see the defaul parameters of litellm's completion parameters: https://docs.litellm.ai/docs/completion/input#input-params-1).
    Except for the following parameters who are overwritten by the ChatAtomicFlow in ChatAtomicFlow.yaml:
        - `model_name` (Union[Dict[str,str],str]): The name of the model to use. 
        When using multiple API providers, the model_name can be a dictionary of the form 
        {"provider_name": "model_name"}. E.g. {"openai": "gpt-3.5-turbo", "azure": "azure/gpt-3.5-turbo"}
        Default: "gpt-3.5-turbo" (the name needs to follow the name of the model in litellm  https://docs.litellm.ai/docs/providers).
        - `n` (int) : The number of answers to generate. Default: 1
        - `max_tokens` (int): The maximum number of tokens to generate. Default: 2000
        - `temperature` float: The temperature of the generation. Default: 0.3
        - `top_p` float: An alternative to sampling with temperature. It instructs the model to consider the results of
        the tokens with top_p probability. Default: 0.2
        - `frequency_penalty` (number): It is used to penalize new tokens based on their frequency in the text so far. Default: 0.0
        - `presence_penalty` (number): It is used to penalize new tokens based on their existence in the text so far. Default: 0.0
        - `stream` (bool): Whether to stream the response or not. Default: True
    - `system_message_prompt_template` (Dict[str,Any]): The template of the system message. It is used to generate the system message. 
    By default its of type aiflows.prompt_template.JinjaPrompt.
    None of the parameters of the prompt are defined by default and therefore need to be defined if one wants to use the system prompt. 
    Default parameters are defined in aiflows.prompt_template.jinja2_prompts.JinjaPrompt.
    - `init_human_message_prompt_template` (Dict[str,Any]): The prompt template of the human/user message used to initialize the conversation
    (first time in). It is used to generate the human message. It's passed as the user message to the LLM.
    By default its of type aiflows.prompt_template.JinjaPrompt. None of the parameters of the prompt are defined by default and therefore need to be defined if one
    wants to use the init_human_message_prompt_template. Default parameters are defined in aiflows.prompt_template.jinja2_prompts.JinjaPrompt.
    - `human_message_prompt_template` (Dict[str,Any]): The prompt template of the human/user message (message used everytime the except the first time in).
    It's passed as the user message to the LLM. By default its of type aiflows.prompt_template.JinjaPrompt and has the following parameters:
        - `template` (str): The template of the human message. Default: see ChatAtomicFlow.yaml for the default value.
        - `input_variables` (List[str]): The input variables of the human message prompt template. Default: ["query"]
    - `previous_messages` (Dict[str,Any]): Defines which previous messages to include in the input of the LLM. Note that if `first_k`and `last_k` are both none,
    all the messages of the flows's history are added to the input of the LLM. Default:
        - `first_k` (int): If defined, adds the first_k earliest messages of the flow's chat history to the input of the LLM. Default: None
        - `last_k` (int): If defined, adds the last_k latest messages of the flow's chat history to the input of the LLM. Default: None
        
    
    *Input Interface Initialized (Expected input the first time in flow)*:
    
    - `query` (str): The query given to the flow. (e.g. {"query":"What's the capital of Switzerland?"})
    
    *Input Interface (Expected input the after the first time in flow)*:
        
    - `query` (str): The query given to the flow. (e.g. {"query": "Are you sure of your answer?"})
        
    *Output Interface*:
    
    - `api_output` (str): The output of the API call. It is the response of the LLM to the input. (e.g. {"api_output": "The Capital of Switzerland is Bern"})
        
    :param system_message_prompt_template: The template of the system message. It is used to generate the system message.
    :type system_message_prompt_template: JinjaPrompt
    :param human_message_prompt_template: The template of the human message. It is used to generate the human message.
    :type human_message_prompt_template: JinjaPrompt
    :param init_human_message_prompt_template: The template of the human message that is used to initialize the conversation (first time in). It is used to generate the human message.
    :type init_human_message_prompt_template: Optional[JinjaPrompt]
    :param backend: The backend of the flow. It is a LLM that is queried via an API. See litellm's supported models and APIs here: https://docs.litellm.ai/docs/providers
    :type backend: LiteLLMBackend
    :param \**kwargs: Additional arguments to pass to the flow. See :class:`aiflows.base_flows.AtomicFlow` for more details.
    """
    REQUIRED_KEYS_CONFIG = ["backend"]

    SUPPORTS_CACHING: bool = True

    system_message_prompt_template: JinjaPrompt
    human_message_prompt_template: JinjaPrompt

    backend: LiteLLMBackend
    init_human_message_prompt_template: Optional[JinjaPrompt] = None

    def __init__(self,
                 system_message_prompt_template,
                 human_message_prompt_template,
                 init_human_message_prompt_template,
                 backend,
                 **kwargs):
        super().__init__(**kwargs)
        self.system_message_prompt_template = system_message_prompt_template
        self.human_message_prompt_template = human_message_prompt_template
        self.init_human_message_prompt_template = init_human_message_prompt_template
        self.backend = backend
        assert self.flow_config["name"] not in [
            "system",
            "user",
            "assistant",
        ], f"Flow name '{self.flow_config['name']}' cannot be 'system', 'user' or 'assistant'"

    def set_up_flow_state(self):
        """ This method sets up the state of the flow and clears the previous messages."""
        super().set_up_flow_state()
        self.flow_state["previous_messages"] = []

    @classmethod
    def _set_up_prompts(cls, config):
        """ This method sets up the prompts of the flow. It instantiates the prompt templates.
        
        :param config: The configuration of the flow.
        :type config: Dict[str, Any]
        :return: The instantiated prompts of the flow.
        :rtype: Dict[str, Any]
        """
        kwargs = {}
        
        kwargs["system_message_prompt_template"] = \
            hydra.utils.instantiate(config['system_message_prompt_template'], _convert_="partial")
        kwargs["init_human_message_prompt_template"] = \
            hydra.utils.instantiate(config['init_human_message_prompt_template'], _convert_="partial")
        kwargs["human_message_prompt_template"] = \
            hydra.utils.instantiate(config['human_message_prompt_template'], _convert_="partial")

        return kwargs
    
    @classmethod
    def _set_up_backend(cls, config):
        """ This method sets up the backend of the flow. It instantiates the backend.
        
        :param config: The configuration of the flow.
        :type config: Dict[str, Any]
        :return: The instantiated backend of the flow.
        :rtype: Dict[str, Any]
        """
        kwargs = {}

        kwargs["backend"] = \
            hydra.utils.instantiate(config['backend'], _convert_="partial")
        
        return kwargs

    @classmethod
    def instantiate_from_config(cls, config):
        """ This method instantiates the flow from a configuration file
        
        :param config: The configuration of the flow.
        :type config: Dict[str, Any]
        :return: The instantiated flow.
        :rtype: ChatAtomicFlow
        """
        flow_config = deepcopy(config)

        kwargs = {"flow_config": flow_config}

        # ~~~ Set up prompts ~~~
        kwargs.update(cls._set_up_prompts(flow_config))
        kwargs.update(cls._set_up_backend(flow_config))

        # ~~~ Instantiate flow ~~~
        return cls(**kwargs)

    def _is_conversation_initialized(self):
        """ This method checks if the conversation is initialized or not.
        
        :return: True if the conversation is initialized, False otherwise.
        :rtype: bool
        """
        if len(self.flow_state["previous_messages"]) > 0:
            return True

        return False

    def get_interface_description(self):
        """ This method returns the description of the flow's input and output interface.
        
        :return: The description of the flow's interface.
        :rtype: Dict[str, Any]
        """
        if self._is_conversation_initialized():

            return {"input": self.flow_config["input_interface_initialized"],
                    "output": self.flow_config["output_interface"]}
        else:
            return {"input": self.flow_config["input_interface_non_initialized"],
                    "output": self.flow_config["output_interface"]}

    @staticmethod
    def _get_message(prompt_template, input_data: Dict[str, Any]):
        """ This method generates a message using a prompt template and input data which should contain the input variables of the prompt template.
        
        :param prompt_template: The prompt template.
        :type prompt_template: JinjaPrompt
        :param input_data: The input data of the prompt template.
        :type input_data: Dict[str, Any]
        :return: The generated message.
        """
        template_kwargs = {}
        for input_variable in prompt_template.input_variables:
            template_kwargs[input_variable] = input_data[input_variable]
        msg_content = prompt_template.format(**template_kwargs)
        return msg_content

    def _add_demonstrations(self, input_data: Dict[str, Any]):
        """ This method adds demonstrations to the flow (If any). The demonstrations should be passed from a DemonstrationFlow
        
        :param input_data: The input data of the flow.
        :type input_data: Dict[str, Any]
        """
        for demonstration in input_data.get("demonstrations",[]):
            query = demonstration["query"]
            response = demonstration["response"]
            
            self._state_update_add_chat_message(content=query,
                                                role=self.flow_config["user_name"])

            self._state_update_add_chat_message(content=response,
                                                role=self.flow_config["assistant_name"])

    def _state_update_add_chat_message(self,
                                       role: str,
                                       content: str) -> None:
        """ This method adds a message to the flow's state.
        
        :param role: The role of the message (e.g. "user", "assistant", "system").
        :type role: str
        :param content: The content of the message.
        :type content: str
        """

        acceptable_roles = [self.flow_config["system_name"],self.flow_config["user_name"],self.flow_config["assistant_name"]]
        if role in acceptable_roles:
            self.flow_state["previous_messages"].append({"role": role , "content": content})
            
        else:
            raise Exception(f"Invalid role: `{role}`.\n"
                            f"Role should be one of: "
                            f"`{acceptable_roles}`, ")
      

        # Log the update to the flow messages list
        chat_message = UpdateMessage_ChatMessage(
            created_by=self.flow_config["name"],
            updated_flow=self.flow_config["name"],
            role=role,
            content=content,
        )
        self._log_message(chat_message)

    def _get_previous_messages(self):
        """ This method returns the previous messages of the flow. If first_k is set, it returns the first k messages. If last_k is set, it returns the last k messages.
        If both are set, it returns the first k messages and the last k messages. If none are set, it returns all the messages.
        
        :return: The previous messages of the flow.
        :rtype: List[Dict[str, Any]]
        """
        all_messages = self.flow_state["previous_messages"]
        first_k = self.flow_config["previous_messages"]["first_k"]
        last_k = self.flow_config["previous_messages"]["last_k"]

        if not first_k and not last_k:
            return all_messages
        elif first_k and last_k:
            return all_messages[:first_k] + all_messages[-last_k:]
        elif first_k:
            return all_messages[:first_k]
        return all_messages[-last_k:]

    def _call(self):
        """ This method calls the backend of the flow (so queries the LLM). It calls the backend with the previous messages of the flow.
        
        :return: The output of the backend.
        :rtype: Any
        """
        messages = self._get_previous_messages()
        _success = False
        attempts = 1
        error = None
        response = None
        while attempts <= self.flow_config['n_api_retries']:
            try:
                response = self.backend(messages=messages,mock_response=False) #set mock_response to True when debugging (fake API request)
                response = [ answer["content"] for answer in response] # because n in the generation parameters can be > 1
                _success = True
                break
            except Exception as e:
                log.error(
                    f"Error {attempts} in calling backend: {e}. "
                    f"Retrying in {self.flow_config['wait_time_between_retries']} seconds..."
                )
                # log.error(
                #     f"The API call raised an exception with the following arguments: "
                #     f"\n{self.flow_state['history'].to_string()}"
                # ) # ToDo: Make this message more user-friendly
                attempts += 1
                time.sleep(self.flow_config['wait_time_between_retries'])
                error = e
        
        if not _success:
            raise error

        return response

    def _initialize_conversation(self, input_data: Dict[str, Any]):
        """ This method initializes the conversation (occurs the first time in). It adds the system message and potentially the demonstrations (if any).
        
        :param input_data: The input data of the flow.
        :type input_data: Dict[str, Any]
        """
        # ~~~ Add the system message ~~~
        system_message_content = self._get_message(self.system_message_prompt_template, input_data)

        self._state_update_add_chat_message(content=system_message_content,
                                            role=self.flow_config["system_name"])

        # # ~~~ Add the demonstration  (if any) ~~~
        self._add_demonstrations(input_data)

    def _process_input(self, input_data: Dict[str, Any]):
        """ This method processes the input of the flow. It adds the human message to the flow's state. If the conversation is not initialized, it also initializes it
        (adding the system message and potentially the demonstrations).
        
        :param input_data: The input data of the flow.
        :type input_data: Dict[str, Any]
        """
        if self._is_conversation_initialized():
            # Construct the message using the human message prompt template
            user_message_content = self._get_message(self.human_message_prompt_template, input_data)

        else:
            # Initialize the conversation (add the system message, and potentially the demonstrations)
            self._initialize_conversation(input_data)
            if getattr(self, "init_human_message_prompt_template", None) is not None:
                # Construct the message using the query message prompt template
                user_message_content = self._get_message(self.init_human_message_prompt_template, input_data)
            else:
                user_message_content = self._get_message(self.human_message_prompt_template, input_data)

        self._state_update_add_chat_message(role=self.flow_config["user_name"],
                                            content=user_message_content)

    def run(self,input_data: Dict[str, Any]):
        """ This method runs the flow. It processes the input, calls the backend and updates the state of the flow.
        
        :param input_data: The input data of the flow.
        :type input_data: Dict[str, Any]
        :return: The LLM's api output.
        :rtype: Dict[str, Any]
        """
        
        # ~~~ Process input ~~~
        self._process_input(input_data)

        # ~~~ Call ~~~
        response = self._call()
        
        #loop is in case there was more than one answer (n>1 in generation parameters)
        for answer in response:
            self._state_update_add_chat_message(
                role=self.flow_config["assistant_name"],
                content=answer
            )
        response = response if len(response) > 1 or len(response) == 0 else response[0]
        return {"api_output": response}
