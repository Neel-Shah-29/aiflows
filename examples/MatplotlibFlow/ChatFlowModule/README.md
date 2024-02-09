---
license: mit
---

# Table of Contents

* [ChatAtomicFlow](#ChatAtomicFlow)
  * [ChatAtomicFlow](#ChatAtomicFlow.ChatAtomicFlow)
    * [set\_up\_flow\_state](#ChatAtomicFlow.ChatAtomicFlow.set_up_flow_state)
    * [instantiate\_from\_config](#ChatAtomicFlow.ChatAtomicFlow.instantiate_from_config)
    * [get\_interface\_description](#ChatAtomicFlow.ChatAtomicFlow.get_interface_description)
    * [run](#ChatAtomicFlow.ChatAtomicFlow.run)

<a id="ChatAtomicFlow"></a>

# ChatAtomicFlow

<a id="ChatAtomicFlow.ChatAtomicFlow"></a>

## ChatAtomicFlow Objects

```python
class ChatAtomicFlow(AtomicFlow)
```

This class implements a ChatAtomicFlow. It is a flow that uses a LLM via an API to generate textuals responses to textual inputs.

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

**Arguments**:

- `system_message_prompt_template` (`JinjaPrompt`): The template of the system message. It is used to generate the system message.
- `human_message_prompt_template` (`JinjaPrompt`): The template of the human message. It is used to generate the human message.
- `init_human_message_prompt_template` (`Optional[JinjaPrompt]`): The template of the human message that is used to initialize the conversation (first time in). It is used to generate the human message.
- `backend` (`LiteLLMBackend`): The backend of the flow. It is a LLM that is queried via an API. See litellm's supported models and APIs here: https://docs.litellm.ai/docs/providers
- `\**kwargs`: Additional arguments to pass to the flow. See :class:`aiflows.base_flows.AtomicFlow` for more details.

<a id="ChatAtomicFlow.ChatAtomicFlow.set_up_flow_state"></a>

#### set\_up\_flow\_state

```python
def set_up_flow_state()
```

This method sets up the state of the flow and clears the previous messages.

<a id="ChatAtomicFlow.ChatAtomicFlow.instantiate_from_config"></a>

#### instantiate\_from\_config

```python
@classmethod
def instantiate_from_config(cls, config)
```

This method instantiates the flow from a configuration file

**Arguments**:

- `config` (`Dict[str, Any]`): The configuration of the flow.

**Returns**:

`ChatAtomicFlow`: The instantiated flow.

<a id="ChatAtomicFlow.ChatAtomicFlow.get_interface_description"></a>

#### get\_interface\_description

```python
def get_interface_description()
```

This method returns the description of the flow's input and output interface.

**Returns**:

`Dict[str, Any]`: The description of the flow's interface.

<a id="ChatAtomicFlow.ChatAtomicFlow.run"></a>

#### run

```python
def run(input_data: Dict[str, Any])
```

This method runs the flow. It processes the input, calls the backend and updates the state of the flow.

**Arguments**:

- `input_data` (`Dict[str, Any]`): The input data of the flow.

**Returns**:

`Dict[str, Any]`: The LLM's api output.

