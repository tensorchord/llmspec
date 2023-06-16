from __future__ import annotations

from enum import Enum
from typing import List

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

    # identifier, will be used to match the model
    name: str

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
    system_msg_template: str = "{content}{sep}"

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
    name="thudm/chatglm",
    user_token="问：",
    assistant_token="答：",
    sep_token="\n",
    user_msg_template="[Round {round}]\n{role}{content}{sep}",
    transformer_model_cls="AutoModel",
    low_cpu_mem_usage=False,
)
MOSS = LanguageModelInfo(
    name="fnlp/moss-moon",
    user_token="<|USER|>",
    assistant_token="<|ASSISTANT|>",
    system_token="<|SYSTEM|>",
    sep_token="\n",
    transformer_model_cls="AutoModelForCausalLM",
    append_assistant_token=True,
)
StableLM = LanguageModelInfo(
    name="stabilityai/stablelm",
    user_token="<|Human|>",
    assistant_token="<|StableLM|>",
    sep_token="\n",
    append_assistant_token=True,
)
LLaMA = LanguageModelInfo(
    name="decapoda-research/llama",
    user_token="USER: ",
    assistant_token="ASSISTANT: ",
    sep_token="\n",
    append_assistant_token=True,
)
Vicuna = LanguageModelInfo(
    name="lmsys/vicuna",
    user_token="USER: ",
    assistant_token="ASSISTANT: ",
    sep_token="\n",
    assistant_sep_token="</s>\n",
    append_assistant_token=True,
)
BloomZ = LanguageModelInfo(
    name="bigscience/bloomz",
    user_token="USER: ",
    assistant_token="ASSISTANT: ",
    sep_token="\n",
    append_assistant_token=True,
)
FastChatT5 = LanguageModelInfo(
    name="lmsys/fastchat",
    user_token="USER: ",
    assistant_token="ASSISTANT: ",
    sep_token="\n### ",
    append_assistant_token=True,
    transformer_model_cls="AutoModelForSeq2SeqLM",
    tokenizer_cls="T5Tokenizer",
)
Falcon = LanguageModelInfo(
    name="tiiuae/falcon",
    user_token="User: ",
    assistant_token="Falcon: ",
    sep_token="\n",
    append_assistant_token=True,
)
Unknown = LanguageModelInfo(
    name="unknown",
)


class LanguageModels(Enum):
    CHAT_GLM = ChatGLM
    MOSS = MOSS
    STABLE_LM = StableLM
    BLOOM_Z = BloomZ
    LLAMA = LLaMA
    VICUNA = Vicuna
    FASTCHATT5 = FastChatT5
    FALCON = Falcon

    UNKNOWN = Unknown

    @classmethod
    def find(cls, name: str) -> LanguageModels:  # noqa: PLR0911
        for item in cls:
            if item.value.name in name.lower():
                return item

        return cls.UNKNOWN

    @classmethod
    def transformer_cls(cls, name: str) -> str:
        return cls.find(name).value.transformer_model_cls
