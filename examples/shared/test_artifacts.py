"""Atomic artifacts and the provenance of the sources of a run."""

from examples.shared.artifacts import PROJECT, checksum, software


def test_software_hashes_the_python_files_of_the_given_directories():
    directory = PROJECT / "examples/shared"

    hashes = software(["examples/shared"])["source_sha256"]

    files = sorted(directory.rglob("*.py"))
    assert sorted(hashes) == sorted(str(p.relative_to(PROJECT)) for p in files)
    assert hashes["examples/shared/artifacts.py"] == checksum(directory / "artifacts.py")
