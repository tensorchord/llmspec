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
    sample: bool = False

    @classmethod
    def from_bytes(cls, buf: bytes):
        return msgspec.json.decode(buf, type=cls)


class PromptCompletionRequest(CompletionRequest):
    model: str
    prompt: str = "<|endoftext|>"
    echo: bool = False
    logprobs: Optional[int] = None
    best_of: int = 1


class ChatCompletionRequest(CompletionRequest):
    model: str
    messages: List[ChatMessage]

    def to_model(self, model: str = "ChatGLM"):
        if model.lower() == "chatglm":
            return dict(
                prompt=self.messages[-1].content,
                history=[msg.content for msg in self.messages[:-1]],
                max_length=self.max_tokens,
                top_p=self.top_p,
                temperature=self.temperature,
            )
        elif model.lower() == "moss":
            prompt = []
            for message in self.messages:
                if message.role == Role.USER:
                    message_prefix = "<|Human|>"
                elif message.role == Role.ASSISTANT:
                    message_prefix = "<|MOSS|>"
                else:
                    message_prefix = ""
                prompt.append(f"{message_prefix} {message.content}")
            prompt = "\n".join(prompt) + "\n<|MOSS|>:"

            # MOSS model parameters
            model_params = {
                "inputs": prompt,
                "do_sample": self.sample,
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
