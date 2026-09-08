import pytest
from pydantic import ValidationError

from src.schemas import BatchRequest


def test_default_mode():
    req = BatchRequest(project="ep", lines=[{"id": "L1", "character_id": "voice", "text": "你好"}])
    assert req.lines[0].mode == "controllable"


@pytest.mark.parametrize(
    "lines",
    [
        [{"id": "../x", "text": "你好", "character_id": "v"}],
        [{"id": "L1", "text": "你好", "mode": "design"}],
        [{"id": "L1", "text": " "}],
        [{"id": "L1", "text": "你好", "character_id": "v"}] * 2,
    ],
)
def test_invalid_batch(lines):
    with pytest.raises(ValidationError):
        BatchRequest(project="ep", lines=lines)
