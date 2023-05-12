import json

from pydantic import BaseModel


class LLMSpec(BaseModel):
    """
    # TODO: docstring
    Example usage
    llmspec = LLMSpec(text="Hello")
    encoded = llmspec.encode()
    print(f"Encoded: {encoded}")

    decoded = llmspec.decode(encoded)
    print(f"Decoded: {decoded}")

    to_model = llmspec.to_model(name="moss")
    print(f"MOSS format: {to_model}")
    """

    text: str

    def encode(self) -> str:
        data = {"text": self.text}
        return json.dumps(data)

    @staticmethod
    def decode(data: str | bytes, encoding: str = "utf-8") -> "LLMSpec":
        if isinstance(data, bytes):
            data = data.decode(encoding)
        decoded_data = json.loads(data)
        return LLMSpec(text=decoded_data)

    def to_model(self, name: str = "moss") -> str:
        if name.lower() == "moss":
            return f": {self.text}<eoh>\n:"
        else:
            raise ValueError(f"Unsupported model: {name}")
