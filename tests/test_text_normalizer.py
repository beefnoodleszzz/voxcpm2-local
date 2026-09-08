import json
from pathlib import Path

import pytest

from src.text_normalizer import integer_reading, normalize_text

FIXTURES = Path(__file__).parent / "fixtures"
CASES = sum(
    (
        json.loads((FIXTURES / name).read_text())
        for name in ("dialogue.json", "numbers.json", "mixed_language.json")
    ),
    [],
)


@pytest.mark.parametrize("case", CASES)
def test_regression(case):
    result = normalize_text(case["input"])
    assert result["normalized"] == case["expected"]
    assert result["original"] == case["input"]


def test_profiles_and_invalid_date():
    assert normalize_text("¥19.9", "narration")["normalized"] == "十九点九元"
    assert normalize_text("¥19.9", "none")["normalized"] == "¥19.9"
    result = normalize_text("2026-02-30")
    assert result["warnings"]
    assert result["normalized"] == "2026-02-30"


@pytest.mark.parametrize(
    ("number", "spoken"),
    [
        (10, "十"),
        (101, "一百零一"),
        (1010, "一千零一十"),
        (100000001, "一亿零一"),
        (10010000, "一千零一万"),
    ],
)
def test_integer(number, spoken):
    assert integer_reading(str(number)) == spoken
