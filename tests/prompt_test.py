import pytest

from llmspec.llmspec import ChatCompletionRequest, ChatMessage, Role


@pytest.fixture
def messages():
    return [
        ChatMessage(role=Role.USER, content="Who are you?"),
        ChatMessage(role=Role.ASSISTANT, content="I'm a bot."),
        ChatMessage(role=Role.USER, content="Do you like English?"),
    ]


@pytest.mark.parametrize(
    "model,messages,expected",
    [
        (
            "thudm/chatglm",
            None,
            (
                "[Round 0]\n"
                "问：Who are you?\n"
                "答：I'm a bot.\n"
                "[Round 1]\n"
                "问：Do you like English?\n"
            ),
        ),
        (
            "lmsys/vicuna",
            None,
            (
                "USER: Who are you?\n"
                "ASSISTANT: I'm a bot.</s>\n"
                "USER: Do you like English?\n"
                "ASSISTANT: "
            ),
        ),
        (
            "lmsys/fastchat",
            None,
            (
                "USER: Who are you?\n### "
                "ASSISTANT: I'm a bot.\n### "
                "USER: Do you like English?\n### "
                "ASSISTANT: "
            ),
        ),
        (
            "decapoda-research/llama",
            None,
            (
                "USER: Who are you?\n"
                "ASSISTANT: I'm a bot.\n"
                "USER: Do you like English?\n"
                "ASSISTANT: "
            ),
        ),
        (
            "tiiuae/falcon-7b-instruct",
            None,
            (
                "User: Who are you?\n"
                "Falcon: I'm a bot.\n"
                "User: Do you like English?\n"
                "Falcon: "
            ),
        ),
        (
            "meta-llama/Llama-2-7b-hf",
            None,
            (
                """<<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as \
possible, while being safe. Your answers should not include any harmful, unethical, \
racist, sexist, toxic, dangerous, or illegal content. Please ensure that your \
responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why \
instead of answering something not correct. If you don't know the answer to a \
question, please don't share false information.
<</SYS>>

"""
                " [INST] Who are you? [/INST] "
                " I'm a bot."
                " [INST] Do you like English? [/INST] "
            )
        )
    ],
    indirect=["messages"],
)
def test_chat_prompt_generation(model, messages, expected):
    chat = ChatCompletionRequest(model=model, messages=messages)
    assert chat.get_prompt(model) == expected
