"""Conservative Chinese spoken-text normalization with an auditable result.

Ambiguous identifiers, URLs and email addresses stay intact and require review.
No pinyin, model-specific markup or pronunciation capability is assumed.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date

from .errors import ConfigurationError

DIGITS = "零一二三四五六七八九"
VERSION = "1"


def digit_reading(value: str) -> str:
    return "".join(DIGITS[int(c)] if c.isdigit() and c.isascii() else c for c in value)


def integer_reading(value: str) -> str:
    n = int(value)
    if n < 0:
        return "负" + integer_reading(str(-n))
    if n == 0:
        return "零"
    if n >= 10**12:
        return digit_reading(value)
    if n >= 10000:
        unit, base = ("亿", 10**8) if n >= 10**8 else ("万", 10000)
        top, rest = divmod(n, base)
        return (
            integer_reading(str(top))
            + unit
            + (("零" if rest < base // 10 else "") + integer_reading(str(rest)) if rest else "")
        )
    result = ""
    pending_zero = False
    for power, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d, n = divmod(n, power)
        if d:
            if pending_zero:
                result += "零"
            result += DIGITS[d] + unit
            pending_zero = False
        elif result and n:
            pending_zero = True
    return result[1:] if result.startswith("一十") else result


def number_reading(value: str) -> str:
    sign = "负" if value.startswith("-") else ""
    value = value.lstrip("+-")
    whole, dot, fractional = value.partition(".")
    return sign + integer_reading(whole) + ("点" + digit_reading(fractional) if dot else "")


def normalize_text(text: str, profile: str = "dialogue") -> dict:
    if profile not in {"none", "dialogue", "narration"}:
        raise ConfigurationError(f"Unknown normalization profile: {profile}")
    if not text.strip():
        raise ConfigurationError("Text cannot be empty")
    original = text
    if profile == "none":
        return {
            "original": original,
            "normalized": text,
            "profile": profile,
            "version": VERSION,
            "warnings": [],
            "changed": False,
        }
    text = unicodedata.normalize("NFKC", text)
    warnings = []
    protected = []

    def protect(match):
        protected.append(match.group())
        warnings.append(f"Review spoken form of identifier: {match.group()}")
        return chr(0xE000 + len(protected) - 1)

    text = re.sub(
        r"https?://[^\s，。！？；]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|[A-Za-z][A-Za-z0-9_.-]*",
        protect,
        text,
    )

    def read_date(match):
        y, m, d = map(int, match.groups())
        try:
            date(y, m, d)
        except ValueError:
            warnings.append(f"Invalid date requires correction: {match.group()}")
            return protect(match)
        return (
            digit_reading(str(y))
            + "年"
            + integer_reading(str(m))
            + "月"
            + integer_reading(str(d))
            + "日"
        )

    text = re.sub(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?", read_date, text)
    text = re.sub(r"(?<!\d)(\d{4})年", lambda m: digit_reading(m[1]) + "年", text)

    def read_time(match):
        h, m, s = match[1], match[2], match[3]
        if int(h) > 23 or int(m) > 59 or (s and int(s) > 59):
            return protect(match)
        value = integer_reading(h) + "点"
        value += ("零" if 0 < int(m) < 10 else "") + integer_reading(m) + "分" if int(m) else "整"
        return value + (integer_reading(s) + "秒" if s else "")

    text = re.sub(r"(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?!\d)", read_time, text)
    text = re.sub(r"(?<!\d)(1[3-9]\d{9})(?!\d)", lambda m: digit_reading(m[1]), text)
    text = re.sub(
        r"(?<!\d)(0\d{2,3})-(\d{7,8})(?!\d)",
        lambda m: digit_reading(m[1]) + "，" + digit_reading(m[2]),
        text,
    )
    number = r"-?\d+(?:\.\d+)?"
    text = re.sub(rf"({number})%", lambda m: "百分之" + number_reading(m[1]), text)
    text = re.sub(
        rf"[¥￥]({number})",
        lambda m: number_reading(m[1]) + ("块" if profile == "dialogue" else "元"),
        text,
    )
    text = re.sub(rf"\$({number})", lambda m: number_reading(m[1]) + "美元", text)
    # Replace operators only between numbers, preserving hyphens in identifiers.
    text = re.sub(
        r"(?<=\d)\s*([+×÷=−])\s*(?=-?\d)",
        lambda m: {"+": "加", "×": "乘以", "÷": "除以", "=": "等于", "−": "减"}[m[1]],
        text,
    )
    text = re.sub(r"(?<=\d)-(?=\d)", "减", text)
    text = re.sub(number, lambda m: number_reading(m[0]), text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("...", "……").replace("~", "到")
    for i, value in enumerate(protected):
        text = text.replace(chr(0xE000 + i), value)
    return {
        "original": original,
        "normalized": text,
        "profile": profile,
        "version": VERSION,
        "warnings": warnings,
        "changed": original != text,
    }
