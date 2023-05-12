import json

from pydantic import BaseModel


class LLMSpec(BaseModel):
    """
    # TODO: docstring
    Example usage
    llm = LLM(text="你好")
    encoded = llm.encode()
    print(f"Encoded: {encoded}")

    decoded = LLM.decode(encoded)
    print(f"Decoded: {decoded}")

    to_model = llm.to_model(name="moss")
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
        return LLMSpec(**decoded_data)

    def to_model(self, name: str = "moss") -> str:
        if name.lower() == "moss":
            return f": {self.text}<eoh>\n:"
        else:
            raise ValueError(f"Unsupported model: {name}")
