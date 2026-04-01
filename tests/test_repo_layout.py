from pathlib import Path


def test_repo_has_core_packages():
    assert Path("src/cope/stage1").is_dir()
    assert Path("src/cope/stage2").is_dir()
    assert Path("src/cope/stage3").is_dir()
    assert Path("src/cope/inference").is_dir()
