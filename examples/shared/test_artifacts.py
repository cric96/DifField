"""Atomic artifacts and the provenance of the sources of a run."""

from examples.shared.artifacts import PROJECT, checksum, csv_write, software


def test_software_hashes_the_python_files_of_the_given_directories():
    directory = PROJECT / "examples/shared"

    hashes = software(["examples/shared"])["source_sha256"]

    files = sorted(directory.rglob("*.py"))
    assert sorted(hashes) == sorted(str(p.relative_to(PROJECT)) for p in files)
    assert hashes["examples/shared/artifacts.py"] == checksum(directory / "artifacts.py")


def test_csv_write_takes_the_columns_in_first_seen_order(tmp_path):
    rows = [{"method": "fixed", "error": 0.5}, {"method": "learned", "seed": 1, "error": None}]

    csv_write(tmp_path / "tables/results.csv", rows)

    lines = (tmp_path / "tables/results.csv").read_text().splitlines()
    assert lines == ["method,error,seed", "fixed,0.5,", "learned,,1"]
