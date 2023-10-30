import json

import pytest
from msgspec import ValidationError

from llmspec import EmbeddingRequest


@pytest.mark.parametrize(
    "text,exception",
    [
        (
            json.dumps({"model": "text-embedding-ada-002", "input": "single sentence"}),
            None,
        ),
        (
            json.dumps({"model": "text-embedding-ada-002", "input": ["list", "list"]}),
            None,
        ),
        (json.dumps({"model": "text-embedding-ada-002", "input": [[0, 233]]}), None),
        (
            json.dumps({"model": "text-embedding-ada-002", "input": [[0, 233], [0]]}),
            None,
        ),
        (
            json.dumps({"model": "text-embedding-ada-002", "input": 233}),
            ValidationError,
        ),
        (json.dumps({"input": "missing-model-name"}), ValidationError),
        (json.dumps({"model": "text-embedding-ada-002"}), ValidationError),
    ],
)
def test_embedding_request(text, exception):
    if exception is None:
        EmbeddingRequest.from_bytes(text)
    else:
        with pytest.raises(exception):
            EmbeddingRequest.from_bytes(text)
