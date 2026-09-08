import hashlib

import pytest

from src.provenance import verified_artifact


def test_artifact_hash_receipt_and_tampering(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"original weights")
    etag = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    assert (
        verified_artifact(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, etag) == etag
    )
    path.write_bytes(b"different weights")
    stat = path.stat()
    with pytest.raises(ValueError):
        verified_artifact(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, etag)
