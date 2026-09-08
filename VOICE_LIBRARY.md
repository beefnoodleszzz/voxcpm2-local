# Production Voice Library

The library contains 24 selected cast voices plus the earlier `male_lead_01` test voice. All 24 cast voices have a canonical `neutral` reference. Six core voices additionally contain `gentle`, `angry`, `sad`, `panic`, and `whisper` references:

- `male_young_sunny_01`
- `male_young_cold_01`
- `male_mature_elite_01`
- `female_young_sweet_01`
- `female_young_cool_01`
- `female_mature_queen_01`

For a registered emotion, use Ultimate mode with that style. For other characters or an unregistered emotion, use controllable mode with `style=neutral` and an `instruct` string. Add a dedicated style reference later if the role becomes recurring.

Production batch input is demonstrated in `configs/episode.production.example.json`.
Each line is written to `<project>/<line_id>/candidate_01.wav` (and additional candidates) with matching JSON metadata; the project manifest records every take.

```bash
.venv/bin/python scripts/batch_dub.py configs/episode.production.example.json
```

Run the non-generative integrity audit at any time:

```bash
.venv/bin/python scripts/audit_voice_library.py
```

Raw audition takes remain under `outputs/raw/voice_cast_v1` and `outputs/raw/emotion_cast_v1`. Canonical references never replace those masters.
