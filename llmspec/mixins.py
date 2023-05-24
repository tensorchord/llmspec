import msgspec


class JSONSerializableMixin:
    def to_json(self):
        return msgspec.json.encode(self)

    @classmethod
    def from_bytes(cls, buf: bytes):
        return msgspec.json.decode(buf, type=cls)
