import pytest

from src.longform import segment_text
from src.speaker_qc import cosine_similarity, speaker_similarity


def test_semantic_segmentation_preserves_text():
    text = "雨已经停了。你还在等谁？我不知道，只想再等一会儿。"
    segments = segment_text(text, 20)
    assert "".join(segments) == text
    assert all(len(segment) <= 20 for segment in segments)


def test_do_not_mechanically_split_unpunctuated_clause():
    with pytest.raises(ValueError):
        segment_text("一" * 30, 20)


def test_closing_quote_stays_with_sentence():
    assert segment_text("他说：“你好。”她没有回答。") == ["他说：“你好。”", "她没有回答。"]


def test_optional_speaker_backend_honest(tmp_path):
    assert speaker_similarity(tmp_path / "a", tmp_path / "b")["status"] == "UNAVAILABLE"
    assert cosine_similarity([1, 0], [1, 0]) == 1
    assert cosine_similarity([1, 0], [0, 1]) == 0
    with pytest.raises(ValueError):
        cosine_similarity([0, 0], [1, 0])
