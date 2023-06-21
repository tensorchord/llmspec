from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import msgspec

from llmspec.mixins import JSONSerializableMixin
from llmspec.model_info import ChatMessage, LanguageModels, Role


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


class ChatFunction(msgspec.Struct):
    name: str
    description: Optional[str]
    parameters: Optional[Dict[str, Any]]


class ChatCompletionRequest(CompletionRequest):
    model: str
    messages: List[ChatMessage]
    functions: Optional[List[ChatFunction]] = None
    # default: "none" if `functions` is empty else "auto"
    function_call: Optional[Union[str, Dict[str, str]]] = None

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
        model: str,
        finish_reason: str,
        prompt_token: int,
        completion_token: int,
        role: Role = Role.ASSISTANT,
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
