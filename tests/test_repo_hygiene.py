"""Guards on the repository itself.

V1 shipped a UTF-16 ``requirements.txt`` (a PowerShell ``pip freeze >``
artifact) that ``pip install -r`` cannot parse. V2 reintroduced it by
overwriting the file in place, which preserved its encoding. Once is an
accident; twice is a missing test.
"""

import subprocess

import pytest

from recommender import config

TEXT_SUFFIXES = {".py", ".txt", ".md", ".toml", ".yml", ".yaml", ".cfg", ".ini"}


def tracked_text_files():
    out = subprocess.run(
        ["git", "ls-files"], cwd=config.ROOT, capture_output=True, text=True, check=True
    )
    return [config.ROOT / line for line in out.stdout.splitlines() if line]


@pytest.mark.parametrize(
    "path",
    [p for p in tracked_text_files() if p.suffix in TEXT_SUFFIXES],
    ids=lambda p: p.name,
)
def test_text_files_are_utf8(path):
    raw = path.read_bytes()
    assert b"\x00" not in raw, f"{path.name} looks like UTF-16"
    raw.decode("utf-8")


def test_requirements_are_parseable():
    for name in ("requirements.txt", "requirements-dev.txt"):
        lines = (config.ROOT / name).read_text(encoding="utf-8").splitlines()
        assert any(line.strip() and not line.startswith("-r") for line in lines)
