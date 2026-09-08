from src.generation_policy import accepted_automatically, plan_attempts, rank_candidates


def test_default_and_experimental_schedule():
    policy, attempts = plan_attempts(profile="production", candidates=1, seed=42, steps=10)
    assert [a["inference_timesteps"] for a in attempts] == [10, 20, 30]
    assert policy["early_stop"]
    policy, attempts = plan_attempts(profile="quality", candidates=3, seed=42, steps=30)
    assert [a["inference_timesteps"] for a in attempts] == [10, 20, 30]
    assert len({a["seed"] for a in attempts}) == 3


def test_missing_or_uncalibrated_asr_never_early_stops():
    assert not accepted_automatically({"qc": {"status": "PASS"}})
    assert not accepted_automatically(
        {"qc": {"status": "PASS"}, "transcript_qc": {"status": "WARN"}}
    )
    assert accepted_automatically({"qc": {"status": "PASS"}, "transcript_qc": {"status": "PASS"}})


def test_failed_audio_cannot_be_recommended():
    result = rank_candidates([{"output": "bad.wav", "qc": {"status": "FAIL"}}])
    assert result["recommended_candidate"] is None
    assert result["human_review_required"]
