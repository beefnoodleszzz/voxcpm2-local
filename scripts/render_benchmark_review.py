"""Generate a local listening page; ratings remain explicit human input."""

import argparse
import html
import json
from pathlib import Path

import _bootstrap  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    args = parser.parse_args()
    report = json.loads(args.benchmark.read_text())
    root = args.benchmark.resolve().parent
    rows = []
    for row in report["rows"]:
        audio = Path(row["output"]).resolve()
        if not audio.is_relative_to(root):
            raise ValueError("Benchmark WAV must be inside its run directory")
        rows.append(
            f"<article><h2>{html.escape(row['case'])} · {row['steps']} steps</h2>"
            f"<p>{html.escape(row['text'])}</p>"
            f"<p>QC: {html.escape(row['qc']['status'])} · CER: {row['cer']} · RTF: {row['rtf']:.2f}</p>"
            f'<audio controls preload="none" src="{html.escape(str(audio.relative_to(root)))}"></audio>'
            f"<details><summary>ASR 与差异</summary><pre>{html.escape(json.dumps(row['transcript_qc'], ensure_ascii=False, indent=2))}</pre></details></article>"
        )
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VoxCPM2 A/B 试听</title><style>body{max-width:980px;margin:32px auto;padding:0 20px;font:16px/1.6 system-ui;color:#202020;background:#fafafa}article{border:1px solid #ccc;padding:16px;margin:16px 0}audio{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>VoxCPM2 固定参考 A/B 试听</h1><p>请分别评价自然度、情绪、角色一致性、发音、停连与瑕疵。CER 不是听感评分，尚未校准 ASR 门槛；生产默认保持 30 steps。</p>'
        + "".join(rows)
        + "</html>"
    )
    output = root / "review.html"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(page)
    print(output)


if __name__ == "__main__":
    main()
