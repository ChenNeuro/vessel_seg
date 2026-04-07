from pathlib import Path

from vessel_seg.config import ProjectPaths


def test_project_paths_resolves_external_asoca_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "data" / "external").mkdir(parents=True)
    pointer = root / "data" / "external" / "asoca2020.path"
    pointer.write_text("/tmp/asoca2020", encoding="utf-8")

    paths = ProjectPaths.from_root(root)

    assert paths.asoca_root == Path("/tmp/asoca2020")
    assert paths.outputs_dir == root / "outputs_reorganized"
