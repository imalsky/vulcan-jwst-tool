"""pandeia_worker backend-identity helpers (no pandeia needed: the worker's
pandeia imports are function-local, so the module imports with numpy alone).

The engine/refdata release-match gate is the STScI rule "engine and refdata
versions must be the same", applied to the leading numeric release segment so
the validated 3.0 + 3.0rc3 pair passes while 2026.1-engine-on-3.0rc3-refdata
(the documented base-env failure mode) is refused BEFORE a deep engine error.
"""
import os

import pytest

from jwst_tool import pandeia_worker as pw


# --- _release ---------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("3.0", "3.0"),
    ("3.0rc3", "3.0"),
    ("2026.2", "2026.2"),
    ("2026.2.dev1", "2026.2"),
    ("  4.1 \n", "4.1"),
    ("rc3", None),
    ("", None),
])
def test_release_segment(raw, expected):
    assert pw._release(raw) == expected


# --- _refdata_version -------------------------------------------------------

def test_refdata_version_prefers_version_file(tmp_path):
    (tmp_path / "VERSION").write_text("2026.2\nextra\n")
    (tmp_path / "VERSION_PSF").write_text("9.9\n")
    ver, src = pw._refdata_version(str(tmp_path))
    assert (ver, src) == ("2026.2", "VERSION")


def test_refdata_version_falls_back_to_psf_then_dirname(tmp_path):
    # the 3.0rc3 tree ships VERSION_PSF only
    tree = tmp_path / "pandeia_data-3.0rc3"
    tree.mkdir()
    (tree / "VERSION_PSF").write_text("3.0\n\nPSF provenance text\n")
    assert pw._refdata_version(str(tree)) == ("3.0", "VERSION_PSF")
    os.remove(tree / "VERSION_PSF")
    ver, src = pw._refdata_version(str(tree))
    assert (ver, src) == ("3.0rc3", "directory name")


def test_refdata_version_undeterminable(tmp_path):
    ver, _src = pw._refdata_version(str(tmp_path))
    assert ver is None


# --- _check_backend_match ---------------------------------------------------

def test_match_accepts_validated_pair(tmp_path):
    tree = tmp_path / "pandeia_data-3.0rc3"
    tree.mkdir()
    (tree / "VERSION_PSF").write_text("3.0\n")
    prov = pw._check_backend_match("3.0", str(tree))
    assert prov == {"refdata_version": "3.0", "refdata_version_source": "VERSION_PSF"}


def test_match_refuses_mismatched_engine(tmp_path):
    tree = tmp_path / "pandeia_data-3.0rc3"
    tree.mkdir()
    (tree / "VERSION_PSF").write_text("3.0\n")
    with pytest.raises(RuntimeError, match="does not match"):
        pw._check_backend_match("2026.1", str(tree))


def test_match_refuses_unidentifiable_refdata(tmp_path):
    with pytest.raises(RuntimeError, match="cannot determine"):
        pw._check_backend_match("3.0", str(tmp_path))
