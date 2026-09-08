import json
from pathlib import Path

import pytest

from src.lexicon import load_lexicon, merge_lexicons
from src.pronunciation import prepare_text, resolve_pronunciation


def test_precedence_and_nonrecursive_longest_match():
    lexicon = merge_lexicons(
        {"重庆": "甲", "重庆银行": "乙"}, {"重庆": "丙"}, {"重庆": "丁", "丁": "戊"}
    )
    assert resolve_pronunciation("重庆银行重庆", lexicon)["text"] == "乙丁"


def test_no_pinyin_claim_or_acronym_substring():
    for case in json.loads((Path(__file__).parent / "fixtures/pronunciation.json").read_text()):
        result = resolve_pronunciation(case["input"], load_lexicon())
        assert result["text"] == case["expected"]
        assert len(result["warnings"]) == case["warnings"]
    assert resolve_pronunciation("MAIL", {"AI": "A I"})["text"] == "MAIL"


def test_expected_text_separate_from_spoken():
    result = prepare_text("小单", "dialogue", {"小单": {"spoken": "小善", "expected": "小单"}})
    assert result["text_spoken"] == "小善"
    assert result["text_expected"] == "小单"


def test_invalid_dictionary():
    with pytest.raises(ValueError):
        merge_lexicons({"重庆": {"spoken": ""}})


def test_string_pinyin_is_hint_not_spoken_latin():
    result = resolve_pronunciation("重庆", {"重庆": "chóng qìng"})
    assert result["text"] == "重庆"
    assert result["warnings"]
