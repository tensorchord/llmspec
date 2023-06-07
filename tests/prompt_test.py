import pytest

from llmspec.llmspec import ChatMessage, Role


@pytest.fixture
def messages():
    return [
        ChatMessage(role=Role.USER, content="Who are you?"),
        ChatMessage(role=Role.ASSISTANT, content="I'm a bot."),
        ChatMessage(role=Role.USER, content="Do you like English?"),
    ]


@pytest.mark.parametrize(
    "messages,expected",
    [],
    indirect=["messages"],
)
def test_prompt_generation(messages, expected):
    pass
