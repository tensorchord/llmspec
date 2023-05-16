import json
from typing import Union, Optional
from enum import Enum


class Role(Enum):
    """Chat roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    @classmethod
    def values(cls):
        return [item.value for item in cls]


class ChatMessage:
    """Unified chat message interface."""

    def __init__(
        self, content: str, role: Role = Role.USER, name: Optional[str] = None
    ) -> None:
        self.content = content
        self.role = role
        self.name = name

        self.validate()

    def validate(self):
        if not self.content or not isinstance(self.content, str):
            raise ValueError("content should be str with length > 0")
        if self.role not in Role.values():
            raise ValueError(f"role should be one of the {Role.values()}")

    def dict(self):
        msg = {"role": self.role, "content": self.content}
        if self.name:
            msg["name"] = self.name
        return msg


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
