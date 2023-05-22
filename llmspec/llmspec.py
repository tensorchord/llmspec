from __future__ import annotations
import json
from typing import Union, Optional, Dict, List
from datetime import datetime
from enum import Enum

import msgspec


class Role(Enum):
    """Chat roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    @classmethod
    def values(cls):
        return [item.value for item in cls]


class ChatMessage(msgspec.Struct):
    """Unified chat message interface."""

    content: str
    role: Role = Role.USER
    name: str = ""


class LanguageModelInfo(msgspec.Struct):
    """Language model information."""

    # token
    user_token: str = ""
    assistant_token: str = ""
    system_token: str = ""
    append_assistant_token: bool = False

    # model class name in `transformers`
    transformer_model_cls: str = "AutoModelForCausalLM"

    def get_conversation_prompt(self, messages: List[ChatMessage]) -> str:
        """Get the prompt for a conversation using the specific tokens of the model."""
        formatted_messages = []
        for message in messages:
            if message.role == Role.USER:
                message_prefix = self.user_token
            elif message.role == Role.ASSISTANT:
                message_prefix = self.assistant_token
            else:
                message_prefix = self.system_token
            formatted_messages.append(f"{message_prefix}{message.content}")
        conversation = "\n".join(formatted_messages)
        if self.append_assistant_token:
            conversation += f"\n{self.assistant_token}"
        return conversation


ChatGLM = LanguageModelInfo(
    user_token="问：",
    assistant_token="答：",
    system_token="",
    transformer_model_cls="AutoModel",
)
MOSS = LanguageModelInfo(
    user_token="<|USER|>",
    assistant_token="<|ASSISTANT|>",
    system_token="<|SYSTEM|>",
    transformer_model_cls="AutoModelForCausalLM",
    append_assistant_token=True,
)
StableLM = LanguageModelInfo(
    user_token="<|Human|>",
    assistant_token="<|StableLM|>",
    system_token="",
    append_assistant_token=True,
)
BloomZ = LanguageModelInfo()
LLaMA = LanguageModelInfo()
Unknown = LanguageModelInfo()


class CompletionRequest(msgspec.Struct, kw_only=True):
    suffix: Optional[str] = None
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[str] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 0.0
    logit_bias: Optional[Dict] = None
    user: str = ""
    do_sample: bool = False

    @classmethod
    def from_bytes(cls, buf: bytes):
        return msgspec.json.decode(buf, type=cls)


class PromptCompletionRequest(CompletionRequest):
    model: str
    prompt: str = "<|endoftext|>"
    echo: bool = False
    logprobs: Optional[int] = None
    best_of: int = 1


class LanguageModels(Enum):
    CHAT_GLM = ChatGLM
    MOSS = MOSS
    STABLE_LM = StableLM
    BLOOM_Z = BloomZ

    UNKNOWN = Unknown

    @classmethod
    def find(cls, name: str) -> LanguageModels:
        if name.lower().startswith("thudm/chatglm"):
            return cls.CHAT_GLM
        if name.lower().startswith("fnlp/moss-moon"):
            return cls.MOSS
        if name.lower().startswith("stabilityai/stablelm"):
            return cls.STABLE_LM
        if name.lower().startswith("bigscience/bloomz"):
            return cls.BLOOM_Z
        return cls.UNKNOWN

    @classmethod
    def transformer_cls(cls, name: str) -> str:
        return cls.find(name).value.transformer_model_cls


class ChatCompletionRequest(CompletionRequest):
    model: str
    messages: List[ChatMessage]

    def get_prompt(self, model: str):
        model = LanguageModels.find(model)
        if model is LanguageModels.CHAT_GLM:
            # ref to https://huggingface.co/THUDM/chatglm-6b/blob/main/modeling_chatglm.py#L1267
            if len(self.messages) == 1:
                return self.messages[0].content
            round = -1
            prompt = ""
            for message in self.messages:
                if message.role == Role.USER:
                    round += 1
                    prompt += f"[Round {round}]\n问：{message.content}\n"
                elif message.role == Role.ASSISTANT:
                    prompt += f"答：{message.content}\n"
                else:
                    prompt += f"{message.content}\n"
            return prompt
        return model.value.get_conversation_prompt(self.messages)

    def get_inference_args(self, model: str):
        if LanguageModels.find(model) == LanguageModels.CHAT_GLM:
            return dict(
                max_length=self.max_tokens,
                top_p=self.top_p,
                temperature=self.temperature,
            )
        elif LanguageModels.find(model) == LanguageModels.MOSS:
            model_params = {
                "do_sample": self.do_sample,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "max_new_tokens": self.max_tokens,
            }

            return model_params

        # return dict by default
        return msgspec.structs.asdict(self)


class ChatChoice(msgspec.Struct):
    message: ChatMessage
    index: int = 0
    finish_reason: str = "stop"


class TokenUsage(msgspec.Struct):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(msgspec.Struct):
    id: str
    object: str
    created: datetime
    choices: List[ChatChoice]
    usage: TokenUsage

    def to_json(self):
        return msgspec.json.encode(self)


class EmbeddingRequest(msgspec.Struct):
    model: str
    input: Union[str, List[str]]
    user: str = ""


class EmbeddingData(msgspec.Struct):
    embedding: List[float]
    index: int
    object: str = "embedding"


class EmbeddingResponse(msgspec.Struct):
    data: EmbeddingData
    model: str
    usage: TokenUsage
    object: str = "list"


class LLMSpec:
    """
    Represents an LLMSpec object.

    Args:
        text (str): The text content of the LLMSpec.

    Examples:
        >>> llmspec = LLMSpec(text="Hello")
        >>> encoded = llmspec.encode()
        >>> print(f"Encoded: {encoded}")
        Encoded: {"text": "Hello"}

        >>> decoded = LLMSpec.decode(encoded)
        >>> print(f"Decoded: {decoded}")
        Decoded: LLMSpec(text='Hello')

        >>> to_model = llmspec.to_model(name="moss")
        >>> print(f"MOSS format: {to_model}")
        MOSS format: : Hello<eoh>\n:
    """

    def __init__(self, text: str):
        self.text = text

    def encode(self) -> str:
        """
        Returns the JSON-encoded representation of the LLMSpec object.

        Returns:
            str: JSON-encoded representation of the LLMSpec object.

        Examples:
            >>> llmspec = LLMSpec(text="Hello")
            >>> encoded = llmspec.encode()
            >>> print(encoded)
            {"text": "Hello"}
        """
        data = {"text": self.text}
        return json.dumps(data)

    @staticmethod
    def decode(data: Union[str, bytes], encoding: str = "utf-8") -> "LLMSpec":
        """
        Decodes the JSON data into an LLMSpec object.

        Args:
            data (Union[str, bytes]): JSON data to be decoded.
            encoding (str, optional): Encoding of the data. Defaults to "utf-8".

        Returns:
            LLMSpec: Decoded LLMSpec object.

        Examples:
            >>> encoded = '{"text": "Hello"}'
            >>> decoded = LLMSpec.decode(encoded)
            >>> print(decoded.text)
            Hello
        """
        if isinstance(data, bytes):
            data = data.decode(encoding)
        decoded_data = json.loads(data)
        return LLMSpec(text=decoded_data)

    def to_model(self, name: str = "moss") -> str:
        """
        Returns the LLMSpec object in a specific model format.

        Args:
            name (str, optional): Model format name. Defaults to "moss".

        Returns:
            str: LLMSpec object in the specified model format.

        Raises:
            ValueError: If an unsupported model format is provided.

        Examples:
            >>> llmspec = LLMSpec(text="Hello")
            >>> moss_model = llmspec.to_model(name="moss")
            >>> print(moss_model)
            : Hello<eoh>\n:
        """
        if name.lower() == "moss":
            return f": {self.text}<eoh>\n:"
        else:
            raise ValueError(f"Unsupported model: {name}")
