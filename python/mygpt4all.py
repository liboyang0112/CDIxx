"""
Python only API for running all GPT4All models.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Union
from gpt4all import pyllmodel

DEFAULT_MODEL_CONFIG = {
    "systemPrompt": "",
    "promptTemplate": "### Human: \n{0}\n### Assistant:\n",
}

ConfigType = Dict[str, str]
MessageType = Dict[str, str]

class GPT4All:
    def __init__(
        self,
        model_name: str,
        model_type: Optional[str] = None,
        n_threads: Optional[int] = None,
        device: Optional[str] = "gpu",
    ):
        """
        Constructor

        Args:
            model_name: Name of GPT4All or custom model. Including ".gguf" file extension is optional but encouraged.
                Default is None, in which case models will be stored in `~/.cache/gpt4all/`.
            model_type: Model architecture. This argument currently does not have any functionality and is just used as
                descriptive identifier for user. Default is None.
            n_threads: number of CPU threads used by GPT4All. Default is None, then the number of threads are determined automatically.
            device: The processing unit on which the GPT4All model will run. It can be set to:
                - "cpu": Model will run on the central processing unit.
                - "gpu": Model will run on the best available graphics processing unit, irrespective of its vendor.
                - "amd", "nvidia", "intel": Model will run on the best available GPU from the specified vendor.
                Alternatively, a specific GPU name can also be provided, and the model will run on the GPU that matches the name if it's available.
                Default is "cpu".

                Note: If a selected GPU device does not have sufficient RAM to accommodate the model, an error will be thrown, and the GPT4All instance will be rendered invalid. It's advised to ensure the device has enough memory before initiating the model.
        """
        self.model_type = model_type
        self.model = pyllmodel.LLModel()
        if device is not None:
            if device != "cpu":
                self.model.init_gpu(model_path=model_name, device=device)
        self.model.load_model(model_name)
        # Set n_threads
        if n_threads is not None:
            self.model.set_thread_count(n_threads)
    def generate(
        self,
        prompt: str,
        chat_session: list[MessageType],
        template: str = "",
        max_tokens: int = 2000,
        temp: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.4,
        repeat_penalty: float = 1.18,
        repeat_last_n: int = 64,
        n_batch: int = 8,
        n_predict: Optional[int] = None,
        streaming: bool = False,
        callback: pyllmodel.ResponseCallbackType = pyllmodel.empty_response_callback,
    ) -> Union[str, Iterable[str]]:
        """
        Generate outputs from any GPT4All model.

        Args:
            prompt: The prompt for the model the complete.
            max_tokens: The maximum number of tokens to generate.
            temp: The model temperature. Larger values increase creativity but decrease factuality.
            top_k: Randomly sample from the top_k most likely tokens at each generation step. Set this to 1 for greedy decoding.
            top_p: Randomly sample at each generation step from the top most likely tokens whose probabilities add up to top_p.
            repeat_penalty: Penalize the model for repetition. Higher values result in less repetition.
            repeat_last_n: How far in the models generation history to apply the repeat penalty.
            n_batch: Number of prompt tokens processed in parallel. Larger values decrease latency but increase resource requirements.
            n_predict: Equivalent to max_tokens, exists for backwards compatibility.
            streaming: If True, this method will instead return a generator that yields tokens as the model generates them.
            callback: A function with arguments token_id:int and response:str, which receives the tokens from the model as they are generated and stops the generation by returning False.

        Returns:
            Either the entire completion or a generator that yields the completion token by token.
        """

        # Preparing the model request
        generate_kwargs: Dict[str, Any] = dict(
            temp=temp,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            repeat_last_n=repeat_last_n,
            n_batch=n_batch,
            n_predict=n_predict if n_predict is not None else max_tokens,
        )

        generate_kwargs["reset_context"] = len(chat_session) == 1
        chat_session.append({"role": "user", "content": prompt})

        prompt = self._format_chat_prompt_template(
            messages=chat_session[-1:],
            default_prompt_header=chat_session[0]["content"]
            if generate_kwargs["reset_context"]
            else "",
            template=template
        )
        # Prepare the callback, process the model response
        output_collector: list[MessageType]
        output_collector = [
            {"content": ""}
        ]

        chat_session.append({"role": "assistant", "content": ""})
        output_collector = chat_session

        def _callback_wrapper(
            callback: pyllmodel.ResponseCallbackType,
            output_collector: list[MessageType],
        ) -> pyllmodel.ResponseCallbackType:
            def _callback(token_id: int, response: str) -> bool:
                nonlocal callback, output_collector

                output_collector[-1]["content"] += response

                return callback(token_id, response)

            return _callback

        # Send the request to the model
        if streaming:
            return self.model.prompt_model_streaming(
                prompt=prompt,
                callback=_callback_wrapper(callback, output_collector),
                **generate_kwargs,
            )

        self.model.prompt_model(
            prompt=prompt,
            callback=_callback_wrapper(callback, output_collector),
            **generate_kwargs,
        )

        return output_collector[-1]["content"]

    def _format_chat_prompt_template(
        self,
        messages: list[MessageType],
        default_prompt_header: str = "",
        default_prompt_footer: str = "",
        template: str = ""
    ) -> str:
        """
        Helper method for building a prompt from list of messages using the self._current_prompt_template as a template for each message.

        Args:
            messages:  list of dictionaries. Each dictionary should have a "role" key
                with value of "system", "assistant", or "user" and a "content" key with a
                string value. Messages are organized such that "system" messages are at top of prompt,
                and "user" and "assistant" messages are displayed in order. Assistant messages get formatted as
                "Response: {content}".

        Returns:
            Formatted prompt.
        """

        if isinstance(default_prompt_header, bool):
            import warnings

            warnings.warn(
                "Using True/False for the 'default_prompt_header' is deprecated. Use a string instead.",
                DeprecationWarning,
            )
            default_prompt_header = ""

        if isinstance(default_prompt_footer, bool):
            import warnings

            warnings.warn(
                "Using True/False for the 'default_prompt_footer' is deprecated. Use a string instead.",
                DeprecationWarning,
            )
            default_prompt_footer = ""

        full_prompt = default_prompt_header + "\n\n" if default_prompt_header != "" else ""

        for message in messages:
            if message["role"] == "user":
                user_message = template.format(message["content"])
                full_prompt += user_message
            if message["role"] == "assistant":
                assistant_message = message["content"] + "\n"
                full_prompt += assistant_message

        full_prompt += "\n\n" + default_prompt_footer if default_prompt_footer != "" else ""

        return full_prompt

def new_session(system_prompt: str = ""):
    return empty_chat_session(system_prompt or DEFAULT_MODEL_CONFIG["systemPrompt"])
def new_template(prompt_template: str = ""):
    return prompt_template or DEFAULT_MODEL_CONFIG["promptTemplate"]

def empty_chat_session(system_prompt: str = "") -> list[MessageType]:
    return [{"role": "system", "content": system_prompt}]
