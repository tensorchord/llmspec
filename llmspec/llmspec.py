from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

import msgspec

from llmspec.mixins import JSONSerializableMixin


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
    user_token: str = "USER: "
    assistant_token: str = "ASSISTANT: "
    system_token: str = ""
    sep_token: str = "\n"
    assistant_sep_token: str = ""
    append_assistant_token: bool = False

    # template
    user_msg_template: str = "{role}{content}{sep}"
    assistant_msg_template: str = "{role}{content}{sep}"
    system_msg_template: str = "{content}{sep_token}"

    # model class name in `transformers`
    transformer_model_cls: str = "AutoModelForCausalLM"
    tokenizer_cls: str = "AutoTokenizer"

    low_cpu_mem_usage: bool = True

    def get_conversation_prompt(self, messages: List[ChatMessage]) -> str:
        """Get the prompt for a conversation using the specific tokens of the model."""
        formatted_messages = []
        round = -1
        for message in messages:
            if message.role == Role.USER:
                round += 1
                msg = self.user_msg_template.format(
                    role=self.user_token,
                    content=message.content,
                    sep=self.sep_token,
                    round=round,
                )
            elif message.role == Role.ASSISTANT:
                msg = self.assistant_msg_template.format(
                    role=self.assistant_token,
                    content=message.content,
                    sep=self.assistant_sep_token or self.sep_token,
                )
            else:
                msg = self.system_msg_template.format(
                    content=message.content,
                    sep=self.sep_token,
                )
            formatted_messages.append(msg)
        conversation = "".join(formatted_messages)
        if self.append_assistant_token:
            conversation += f"{self.assistant_token}"
        return conversation


# ref to https://huggingface.co/THUDM/chatglm-6b/blob/main/modeling_chatglm.py#L1267
ChatGLM = LanguageModelInfo(
    user_token="问：",
    assistant_token="答：",
    system_token="",
    sep_token="\n",
    user_msg_template="[Round {round}]\n{role}{content}{sep}",
    transformer_model_cls="AutoModel",
    low_cpu_mem_usage=False,
)
MOSS = LanguageModelInfo(
    user_token="<|USER|>",
    assistant_token="<|ASSISTANT|>",
    system_token="<|SYSTEM|>",
    sep_token="\n",
    transformer_model_cls="AutoModelForCausalLM",
    append_assistant_token=True,
)
StableLM = LanguageModelInfo(
    user_token="<|Human|>",
    assistant_token="<|StableLM|>",
    system_token="",
    sep_token="\n",
    append_assistant_token=True,
)
LLaMA = LanguageModelInfo(
    user_token="USER: ",
    assistant_token="ASSISTANT: ",
    system_token="",
    sep_token="\n",
    append_assistant_token=True,
)
Vicuna = LanguageModelInfo(
    user_token="USER: ",
    assistant_token="ASSISTANT: ",
    system_token="",
    sep_token="\n",
    assistant_sep_token="</s>\n",
    append_assistant_token=True,
)
BloomZ = LanguageModelInfo(
    user_token="USER: ",
    assistant_token="ASSISTANT: ",
    system_token="",
    sep_token="\n",
    append_assistant_token=True,
)
FastChatT5 = LanguageModelInfo(
    user_token="USER: ",
    assistant_token="ASSISTANT: ",
    system_token="",
    sep_token="\n### ",
    append_assistant_token=True,
    transformer_model_cls="AutoModelForSeq2SeqLM",
    tokenizer_cls="T5Tokenizer",
)
Unknown = LanguageModelInfo()


class CompletionRequest(msgspec.Struct, JSONSerializableMixin, kw_only=True):
    suffix: Optional[str] = None
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 1
    n: int = 1
    stream: bool = False
    stop: Optional[Union[str, List[int]]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 0.0
    logit_bias: Optional[Dict] = None
    user: str = ""
    do_sample: bool = False


class PromptCompletionRequest(CompletionRequest):
    model: str
    prompt: Union[str, List[str]] = "<|endoftext|>"
    echo: bool = False
    logprobs: Optional[int] = None
    best_of: int = 1

    def get_prompt(self):
        if isinstance(self.prompt, list):
            return "\n".join(self.prompt)
        return self.prompt


class LanguageModels(Enum):
    CHAT_GLM = ChatGLM
    MOSS = MOSS
    STABLE_LM = StableLM
    BLOOM_Z = BloomZ
    LLAMA = LLaMA
    VICUNA = Vicuna
    FASTCHATT5 = FastChatT5

    UNKNOWN = Unknown

    @classmethod
    def find(cls, name: str) -> LanguageModels:  # noqa: PLR0911
        if name.lower().startswith("thudm/chatglm"):
            return cls.CHAT_GLM
        if name.lower().startswith("fnlp/moss-moon"):
            return cls.MOSS
        if name.lower().startswith("stabilityai/stablelm"):
            return cls.STABLE_LM
        if name.lower().startswith("bigscience/bloomz"):
            return cls.BLOOM_Z
        if name.lower().startswith("decapoda-research/llama"):
            return cls.LLAMA
        if name.lower().startswith("lmsys/vicuna"):
            return cls.VICUNA
        if name.lower().startswith("lmsys/fastchat"):
            return cls.FASTCHATT5
        return cls.UNKNOWN

    @classmethod
    def transformer_cls(cls, name: str) -> str:
        return cls.find(name).value.transformer_model_cls


class ChatCompletionRequest(CompletionRequest):
    model: str
    messages: List[ChatMessage]

    def get_prompt(self, model: str):
        language_model = LanguageModels.find(model)
        return language_model.value.get_conversation_prompt(self.messages)

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
    finish_reason: Optional[str] = None


class TokenUsage(msgspec.Struct):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(msgspec.Struct):
    text: str
    index: int = 0
    logprobs: Optional[int] = None
    finish_reason: Optional[str] = None


class LMResponse(msgspec.Struct, JSONSerializableMixin):
    id: str
    object: str
    created: datetime
    model: str
    usage: TokenUsage


class CompletionResponse(LMResponse):
    choices: List[CompletionChoice]

    @classmethod
    def from_message(
        cls,
        message: str,
        model: str,
        finish_reason: str,
        prompt_token: int,
        completion_token: int,
    ):
        return cls(
            id=str(uuid.uuid4()),
            object="completion",
            created=datetime.now(),
            model=model,
            choices=[
                CompletionChoice(
                    text=message,
                    finish_reason=finish_reason,
                )
            ],
            usage=TokenUsage(
                prompt_tokens=prompt_token,
                completion_tokens=completion_token,
                total_tokens=prompt_token + completion_token,
            ),
        )


class ChatResponse(LMResponse):
    choices: List[ChatChoice]

    @classmethod
    def from_message(
        cls,
        message: str,
        role: Role,
        model: str,
        finish_reason: str,
        prompt_token: int,
        completion_token: int,
    ):
        return cls(
            id=str(uuid.uuid4()),
            object="chat",
            created=datetime.now(),
            model=model,
            choices=[
                ChatChoice(
                    message=ChatMessage(
                        content=message,
                        role=role,
                    ),
                    finish_reason=finish_reason,
                ),
            ],
            usage=TokenUsage(
                prompt_tokens=prompt_token,
                completion_tokens=completion_token,
                total_tokens=prompt_token + completion_token,
            ),
        )


class EmbeddingRequest(msgspec.Struct, JSONSerializableMixin):
    model: str = ""
    input: Union[str, List[str]] = None
    user: str = ""
    encoding_format: str = "json"


class EmbeddingData(msgspec.Struct):
    embedding: Union[List[float], str]
    index: int
    object: str = "embedding"


class EmbeddingResponse(msgspec.Struct, JSONSerializableMixin):
    data: List[EmbeddingData]
    model: str
    usage: TokenUsage
    object: str = "list"


class ErrorMessage(msgspec.Struct):
    code: int
    type: str
    message: str
    param: str


class ErrorResponse(msgspec.Struct, JSONSerializableMixin):
    error: ErrorMessage

    @classmethod
    def from_validation_err(
        cls, err: msgspec.ValidationError, param: str = ""
    ) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorMessage(
                code=400,
                type="validation_error",
                message=str(err),
                param=param,
            )
        )
