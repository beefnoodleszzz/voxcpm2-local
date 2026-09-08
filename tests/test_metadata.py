from src.provenance import fingerprint, read_model_lock, runtime_versions


def test_hash_order_stable_content_sensitive():
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"seed": 1}) != fingerprint({"seed": 2})


def test_lock_is_exact_and_runtime_recorded():
    lock = read_model_lock()
    assert len(lock["revision"]) == 40
    assert lock["format"] == "bf16"
    assert runtime_versions()["python_version"]
