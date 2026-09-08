from __future__ import annotations

import re

from .lexicon import load_lexicon, merge_lexicons
from .provenance import fingerprint
from .text_normalizer import normalize_text


def resolve_pronunciation(text: str, lexicon: dict) -> dict:
    lexicon = merge_lexicons(lexicon)
    applied, warnings = [], []
    if not lexicon:
        return {
            "text": text,
            "expected": text,
            "applied": [],
            "warnings": [],
            "sha256": fingerprint(lexicon),
        }
    pattern = re.compile(
        "|".join(
            (r"(?<![A-Za-z])" + re.escape(key) + r"(?![A-Za-z])")
            if key.isascii()
            else re.escape(key)
            for key in sorted(lexicon, key=len, reverse=True)
        )
    )

    def replace(match, expected=False):
        key = match.group()
        value = lexicon[key]
        if isinstance(value, str):
            value = {"spoken": value}
        spoken = value.get("spoken", key)
        if not expected:
            applied.append({"term": key, **value})
            if "pinyin" in value and "spoken" not in value:
                warnings.append(
                    f"{key}: pronunciation hint {value['pinyin']} requires listening; model has no verified phoneme input"
                )
        return value.get("expected", spoken) if expected else spoken

    spoken = pattern.sub(replace, text)
    expected = pattern.sub(lambda match: replace(match, True), text)
    return {
        "text": spoken,
        "expected": expected,
        "applied": applied,
        "warnings": warnings,
        "sha256": fingerprint(lexicon),
    }


def prepare_text(
    text: str,
    normalization_profile: str = "none",
    project_lexicon: dict | None = None,
    episode_lexicon: dict | None = None,
) -> dict:
    normal = normalize_text(text, normalization_profile)
    # Legacy direct generation is unchanged unless preprocessing is explicitly selected.
    lexicon = merge_lexicons(
        load_lexicon() if normalization_profile != "none" else {}, project_lexicon, episode_lexicon
    )
    resolved = resolve_pronunciation(normal["normalized"], lexicon)
    return {
        "text_original": text,
        "text_normalized": normal["normalized"],
        "text_spoken": resolved["text"],
        "text_expected": resolved["expected"],
        "normalization": normal,
        "pronunciation": resolved,
    }
