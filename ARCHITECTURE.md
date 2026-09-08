# Local workstation architecture

The existing Python/MLX engine, API, Gradio UI and voice folders remain the foundation.
There is no database, remote audio service, distributed scheduler or second production TTS model.

```text
source text / episode
  → preparation: emotion route, normalization, layered pronunciation dictionary
  → generation policy: production or explicit experimental bounded schedule
  → serialized BF16 VoxCPM2
  → immutable 48kHz mono PCM-24 candidate and reproducibility metadata
  → objective audio QC + optional offline ASR / character alignment
  → next scheduled candidate or calibrated early acceptance
  → candidate recommendation and human A/B review
  → approved copy with review record
  → optional mastering / approved-segment assembly
  → separate delivery output
```

## Boundaries

`text_normalizer`, `lexicon`, `pronunciation` and `emotions` prepare model inputs without loading
MLX. Original, normalized, spoken and expected text stay separate. A spoken homophone override
must not silently replace the expected transcript. Episode overrides beat project overrides,
which beat `configs/pronunciation.json`. Pinyin hints stay advisory.

`generation_policy` plans a finite number of seeds/steps. Named profiles define their full
schedule; explicit candidate/step fields apply to the legacy/custom path. `engine` owns the
process singleton, in-process inference lock and a cross-process BF16 ownership lease. Loading
a second BF16 process fails visibly. MLX import is lazy so logic tests do not require Metal.

`provenance` fixes the current Hugging Face revision and validates artifact receipts and content
hashes. Large-file checks are cached only while size and filesystem change timestamps match.
Metadata records versions, machine, text layers, reference and transcript hashes, generation
parameters, timing, RTF, memory, audio hash, QC and ASR identity. Reproducibility is traceability
to these inputs; bit-exact output is not promised across different software/hardware versions.

`transcript_qc` launches a bounded local argv subprocess, never a shell command and never an
online fallback. It supplies only the WAV path, not the expected text, preventing prompt leakage
into the recognition score. The independent ASR process releases memory on exit. ASR errors are
recorded explicitly without deleting the completed raw WAV. Speaker embeddings have a separate
optional adapter; no embedding model is automatically installed.

## Persistence and resume

`batch` validates all voice paths and inputs before expensive generation. Its fingerprint includes
the entire request, prepared text, reference audio/transcript/voice metadata, model revision and
artifact hashes, runtime versions, profile, ASR configuration and source-module hashes.

Each project is locked while running. Matching revision manifests and candidate hashes permit
reuse. Every completed line is checkpointed with atomic JSON replacement. A completed WAV whose
metadata was never written is an orphan and causes a fresh revision. A mismatched or corrupt
revision is retained. Candidate writes use exclusive file creation; manifests are derived mutable
indexes. Failed attempts have separate error records. Resume never trusts file existence alone.

```text
outputs/
  raw/<project>/run_0001/<line>/candidate_01.wav + .json
  raw/<project>/run_0001/manifest.json
  raw/<project>/run_0001/<line>/selection.json
  approved/<raw-parent>/<review-id>/candidate_01.wav + .json
  delivery/<approved-parent>/<delivery-id>/candidate_01.wav + .json
  qc/events.jsonl
  qc/reviews/...
  qc/audits/...
  qc/voice_metadata/...   # backups before additive metadata enrichment
```

`review` rechecks candidate integrity, records a named listening decision and copies the approved
audio without modifying raw. `mastering` accepts only approved audio and creates a fresh delivery.
Default delivery is byte-identical. Linear gain respects headroom and reports unmet LUFS targets
instead of silently compressing. `longform` preserves punctuation boundaries and joins only
ordered approved segments; joined performance still needs listening.

## Failure semantics

Configuration, missing voices, invalid references, generation errors, invalid audio, ASR errors
and OOM have distinct error classes. Invalid PCM input is rejected before lossy clipping could
occur. OOM is visible, logged and suggests shorter text / fewer attempts / releasing memory;
there is no automatic precision change. API generation errors return a typed 503 response.

QC is `PASS`, `WARN` or `FAIL`; unavailable/error ASR states are explicit. Numeric acceptance
never promotes audio to approved. Fixed retry budgets prevent ASR mistakes causing endless runs.
