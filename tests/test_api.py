from fastapi.testclient import TestClient
import src.server as server
from src.server import app

def test_health():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["backend"] == "mlx"
    assert r.json()["model_format"] == "BF16"
    assert r.json()["sample_rate"] == 48000

def test_design_validation():
    r = TestClient(app).post("/v1/tts/design", json={"text": "你好", "instruct": ""})
    assert r.status_code == 422


def test_batch_honors_candidate_count(monkeypatch, tmp_path):
    calls = []
    def fake_generate(method, folder, **kwargs):
        folder.mkdir(parents=True, exist_ok=True)
        calls.append((method, folder, kwargs))
        return [{"output": str(folder / f"candidate_{i:02d}.wav")} for i in range(1, kwargs["candidates"] + 1)]
    monkeypatch.setattr(server, "ROOT", tmp_path)
    monkeypatch.setattr(server.engine, "generate_candidates", fake_generate)
    response = TestClient(app).post("/v1/tts/batch", json={
        "project": "ep_test", "candidates": 3,
        "lines": [{"id": "S01_L01", "character_id": "male_young_cold_01",
                   "style": "angry", "text": "测试台词", "mode": "ultimate"}]
    })
    assert response.status_code == 200
    assert len(response.json()["lines"][0]["candidates"]) == 3
    assert calls[0][2]["candidates"] == 3
