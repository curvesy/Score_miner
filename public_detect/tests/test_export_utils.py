from pathlib import Path

from public_detect.export_utils import check_size_gate, copy_export_to_repo, directory_size_bytes


def test_size_gate_counts_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "weights.onnx").write_bytes(b"a" * 10)
    (repo / "README.md").write_text("hello")

    report = check_size_gate(repo, max_mb=1)

    assert report["passes"] is True
    assert report["size_bytes"] == 15
    assert directory_size_bytes(repo) == 15


def test_size_gate_fails_over_limit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "weights.onnx").write_bytes(b"a" * 11)

    report = check_size_gate(repo, max_mb=0.00001)

    assert report["passes"] is False


def test_copy_export_to_repo_uses_standard_name(tmp_path: Path) -> None:
    source = tmp_path / "model.onnx"
    source.write_bytes(b"onnx")

    target = copy_export_to_repo(source, tmp_path / "repo")

    assert target.name == "weights.onnx"
    assert target.read_bytes() == b"onnx"
