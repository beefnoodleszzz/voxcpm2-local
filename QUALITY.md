# Quality reference

Priority: naturalness, character identity, emotion, pronunciation, prosody, reproducibility,
stability, then speed and memory. BF16 and mlx-audio==0.5.1 remain production defaults.

## Generation profiles

| Profile | Steps | Early stop | Status |
| --- | --- | --- | --- |
| production | 10 / 20 / 30 | Yes, when audio and calibrated ASR pass | Approved 2026-09-08 |
| fast | 10 | Yes, if calibrated QC passes | Experimental, opt-in |
| balanced | 10 / 20 | Yes, if calibrated QC passes | Experimental, opt-in |
| quality | 10 / 20 / 30 | Yes, if calibrated QC passes | Experimental, opt-in |
| ultimate | 30 / 30 / 30 | No | Human review required |

Profile names control candidate strategy, not clone mode. `mode=ultimate` remains an explicit
performance-continuation choice. Ordinary dialogue uses neutral reference + controllable mode +
an acting instruction. Direct API/CLI calls without a named profile preserve explicit legacy
steps/candidate settings. Preparation and the UI select `production` by default. In routine use,
only the 10-step candidate is generated; 20 and 30 are fallback attempts after a failed gate.

Every scheduled candidate is retained. Acoustic failures use another seed/scheduled steps.
Transcript mismatches are labelled for normalization/lexicon review; the system does not invent
text corrections or guarantee that higher steps repair words. Emotion and identity failures
require listening/reference/instruction decisions. No retry budget can exceed five attempts.

## Transcript verification

The optional [SenseVoiceSmall MLX model](https://huggingface.co/mlx-community/SenseVoiceSmall)
is pinned in `configs/asr.model.lock.json`. Install with `scripts/download_asr.py`.
`configs/asr.json` selects its offline subprocess adapter. Set `command` to `null` to disable ASR.
Missing weights produce `ERROR`, not a passing score; disabled ASR produces `UNAVAILABLE`.

Alignment reports expected/recognized text, normalized comparison strings, CER, substitutions,
deletions and insertions with positions. CER = edit distance / expected character count, and
can exceed 1 with many insertions. Punctuation and case are ignored; common numeric forms are
normalized symmetrically. Homophones and ASR hallucinations cannot be diagnosed reliably from
CER alone. Empty recognition is full deletion. No expected-text prompt is sent to ASR.

The calibrated production threshold is CER ≤ 0.12. It accepts every human-approved 10-step take;
across all 30 approved samples it creates one conservative false retry (3.33%), the 20-step
mixed-language outlier at CER 0.2105. Adaptive early acceptance requires audio `PASS` and ASR
`PASS`; it never grants human approval.

Build and fill a calibration listening sheet:

```bash
.venv/bin/python scripts/calibrate_asr.py outputs/raw/benchmark_suite/RUN/benchmark.json calibration.samples.json
# Listen; fill human_transcript, listened, reviewer and tts_correct for each sample.
.venv/bin/python scripts/calibrate_asr.py calibration.samples.json calibration.result.json \
  --threshold CHOSEN_VALUE --rationale "Explain observed ASR errors and acceptable false-retry rate"
```

At least 30 distinct, human-reviewed, correctly spoken samples from one backend are required.
The human transcript represents actual speech, independently of the intended TTS text. This
separates ASR's own mistakes from TTS errors. The output records the dataset hash, CER distribution,
review count and observed false-retry rate. Review that evidence before copying the result into
the `calibration` object in `configs/asr.json`; runtime also checks backend identity and sample count.

## Audio and reference QC

Hard failures include wrong sample rate/channel count, wrong raw subtype, empty audio, NaN/Inf,
clipping and silence-only output. Floating output beyond PCM full scale is rejected before PCM
encoding. Warning heuristics cover low energy, near-full-scale peaks, DC offset, silence ratio,
edge/long silence, suspicious duration and large sample jumps. These are engineering heuristics,
not perceptually calibrated scores. They are reported in `qc.py` with an explicit version.

Reference files require safe audio/transcript paths and nonempty transcripts. Existing short
references are warnings, not automatically replaced. Prefer clean mono speech around 5–15 seconds,
especially 8–15 seconds, one stable performance, precise transcript, no clipping and comfortable
level. Noise, music, room/reverb quality and mixed emotions still require listening; the system
explicitly lists these as unmeasured rather than guessing from RMS.

## Pronunciation and normalization

`none` preserves legacy input. `dialogue` and `narration` normalize numbers, decimal/currency,
percentages, dates, times, phones and basic arithmetic while preserving ambiguous identifiers,
URLs and email for explicit review/lexicon decisions. For example ¥19.9 becomes 十九点九块 in
dialogue and 十九点九元 in narration. Long or ambiguous domain-specific expressions require an
explicit tested dictionary entry. Normalization is deliberately not claimed to cover all language.

Dictionary precedence is episode > project > global. Replacements use longest match once, so
one replacement cannot recursively trigger another. `{"spoken":"...","expected":"..."}`
separates model input from transcript comparison. Pinyin-only entries, including tone-marked
string hints, are shown for review rather than injected as Latin text: no verified phoneme API
exists in the production adapter. High name/polyphone accuracy remains a listening acceptance item.

## Quality benchmark and candidate selection

`benchmark_suite/cases.json` covers ordinary speech, anger, restrained sadness, panic, whisper,
a long sentence, numbers/date, mixed language, polyphones and names/places. The same character,
neutral reference, acting instruction and seed are compared at 10/20/30 steps. All audio and
metadata are retained in a unique run. `benchmark_suite/consistency.json` holds a common sentence
across neutral/gentle/angry/sad/panic/whisper for within-character listening.

```bash
.venv/bin/python scripts/benchmark_quality.py
.venv/bin/python scripts/render_benchmark_review.py outputs/raw/benchmark_suite/RUN/benchmark.json
```

Rows include generation time, duration, RTF, peak/RMS, memory, ASR edits and QC. Human naturalness,
emotion, identity, pronunciation, prosody and artifact ratings remain null until listened.
Speaker similarity remains null unless an independently validated embedding backend is supplied
to the optional `speaker_qc` adapter. Its cosine score is advisory, not an identity guarantee.

Recommendation ranks hard audio validity first, then transcript status, warning-free audio and
CER. It is a transparent ordering, not a fabricated weighted naturalness score. All hard-failed
audio yields no recommendation. Humans may choose any valid take after documenting unresolved
warnings. The user passed all 30 takes by ear without an obvious step-count difference. The
measurements, listening verdict and production choice are recorded in `benchmark_suite/decision.json`.
Re-run a broader fixed suite before changing this decision after a model, voice or genre change.

## Delivery loudness

`raw_master`, `drama_dialogue`, `narration` and `short_video` are available use-case labels with
processing off by default. They intentionally have no unvalidated common LUFS target.
[EBU R 128](https://tech.ebu.ch/publications/r128) specifies average programme loudness of
−23 LUFS; this is not a universal target for an isolated dialogue stem or a short-video voice.
Select delivery requirements and listen before adopting any target.

Optional processing trims only outer silence with padding, or applies linear gain. LUFS uses
local FFmpeg measurement and headroom constraints, then remeasures the output; an unachievable
target is reported. Large boosts are refused. No default limiter, EQ, compression or denoising
is added. Processing creates a new delivery and preserves approved and raw audio.

## Acceptance boundaries

Pure tests require no weights/Metal. Real BF16 smoke, endpoint/batch checks and benchmark are
separate local operations. A byte-identical smoke comparison proves that fixed sample's regression
result, not universal emotional or speaker consistency. Calibration, listening scores, polyphone
accuracy and production-profile adoption require human review. Optional speaker models, P2 UI
polish, 8-bit comparison and LoRA experiments remain outside the current validated production path.
