from __future__ import annotations

import json
import re
from pathlib import Path

from .errors import ConfigurationError

GLOBAL_PATH = Path(__file__).resolve().parents[1] / "configs/pronunciation.json"


def load_lexicon(path: Path | None = None) -> dict:
    return json.loads((path or GLOBAL_PATH).read_text(encoding="utf-8"))


def merge_lexicons(
    global_lexicon: dict, project: dict | None = None, episode: dict | None = None
) -> dict:
    merged = {**global_lexicon, **(project or {}), **(episode or {})}
    for key, value in merged.items():
        if not key or not isinstance(key, str) or not isinstance(value, (str, dict)):
            raise ConfigurationError(
                "Lexicon entries need a nonempty key and a string/object value"
            )
        if isinstance(value, dict):
            if not {"spoken", "pinyin"} & value.keys():
                raise ConfigurationError(f"Lexicon {key}: spoken or pinyin required")
            for field in ("spoken", "expected", "pinyin"):
                if field in value and (
                    not isinstance(value[field], str) or not value[field].strip()
                ):
                    raise ConfigurationError(f"Lexicon {key}: invalid {field}")
        elif not value.strip():
            raise ConfigurationError(f"Lexicon {key}: empty replacement")
        elif re.search(r"[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]", value) and re.fullmatch(
            r"[a-zA-Züāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ\s'-]+", value
        ):
            merged[key] = {"pinyin": value}
    return merged
