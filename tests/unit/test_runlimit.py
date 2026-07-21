"""The v18.1 public-instance run limiter: slot semantics + lifecycle."""
import pytest

from jwst_tool import runlimit


@pytest.fixture(autouse=True)
def _tmp_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(runlimit, "SLOT_DIR", tmp_path / "run_slots")


def test_slots_cap_and_release():
    s1 = runlimit.acquire("a")
    s2 = runlimit.acquire("b")
    assert s1 is not None and s2 is not None
    assert runlimit.acquire("c") is None          # cap reached
    assert runlimit.busy_count() == 2
    s1.release()
    s3 = runlimit.acquire("d")                    # freed slot reusable
    assert s3 is not None
    s2.release()
    s3.release()
    assert runlimit.busy_count() == 0


def test_slot_files_persist_and_carry_metadata():
    import json
    s = runlimit.acquire("tagged")
    p = runlimit.SLOT_DIR / f"slot{s.index}.lock"
    meta = json.loads(p.read_text())
    assert meta["tag"] == "tagged" and meta["pid"] > 0
    s.release()
    # lifecycle contract: the slot FILE persists after release (unlinking a
    # flock'd path is the two-inode double-lock race)
    assert p.is_file()
    s.release()                                    # idempotent
