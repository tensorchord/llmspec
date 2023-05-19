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


class RoleTokenMap(msgspec.Struct, kw_only=True):
    user_token: str = ""
    assistant_token: str = ""
    system_token: str = ""


class ChatGLMRoleTokenMap(RoleTokenMap):
    """
    ChatGLM role token map.
    Currently not used because standard ChatGLM prompt
    requires the inclusion of a dynamic round number.
    """

    user_token: str = "问："
    assistant_token: str = "答："
    system_token: str = ""


class MOSSRoleTokenMap(RoleTokenMap):
    """
    MOSS role token map.
    """

    user_token: str = "<|USER|>"
    assistant_token: str = "<|ASSISTANT|>"
    system_token: str = "<|SYSTEM|>"


class StableLMRoleTokenMap(RoleTokenMap):
    """
    StableLM role token map.
    """

    user_token: str = "<|Human|>"
    assistant_token: str = "<|StableLM|>"
    system_token: str = ""


def get_standard_conversation_prompt(
    messages: List[ChatMessage],
    role_token_map: RoleTokenMap,
    append_assistant_token: bool = True,
) -> str:
    """Get a prompt for a conversation using the commonly used format in chat models."""
    formatted_messages = []
    for message in messages:
        if message.role == Role.USER:
            message_prefix = role_token_map.user_token
        elif message.role == Role.ASSISTANT:
            message_prefix = role_token_map.assistant_token
        else:
            message_prefix = role_token_map.system_token
        formatted_messages.append(f"{message_prefix}{message.content}")
    prompt_suffix = (
        f"\n{role_token_map.assistant_token}" if append_assistant_token else ""
    )
    return "\n".join(formatted_messages) + prompt_suffix


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
    CHAT_GLM = "ChatGLM"
    MOSS = "MOSS"
    STABLE_LM = "StableLM"

    UNKNOWN = "unknown"

    @classmethod
    def find(cls, name: str) -> "LanguageModels":
        if name.lower().startswith("thudm/chatglm"):
            return cls.CHAT_GLM
        if name.lower().startswith("fnlp/moss-moon"):
            return cls.MOSS
        if name.lower().startswith("stabilityai/stablelm"):
            return cls.STABLE_LM
        return cls.UNKNOWN


class ChatCompletionRequest(CompletionRequest):
    model: str
    messages: List[ChatMessage]

    def get_prompt(self, model: str):
        if LanguageModels.find(model) == LanguageModels.CHAT_GLM:
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
        elif LanguageModels.find(model) == LanguageModels.MOSS:
            return get_standard_conversation_prompt(
                self.messages, MOSSRoleTokenMap(), True
            )
        elif LanguageModels.find(model) == LanguageModels.STABLE_LM:
            return get_standard_conversation_prompt(
                self.messages, StableLMRoleTokenMap(), True
            )

        # return all the content by default
        return get_standard_conversation_prompt(self.messages, RoleTokenMap(), False)

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
