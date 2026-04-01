from unittest.mock import patch

import train


def test_train_cli_routes_stage2(monkeypatch):
    called = {"stage2": False}

    def fake_stage2(config_path):
        called["stage2"] = config_path.endswith(".yaml")

    monkeypatch.setattr(train, "train_stage2", fake_stage2)
    monkeypatch.setattr(train, "emit_run_metadata", lambda *a, **k: None)
    monkeypatch.setattr(train, "_output_dir_from_config", lambda _: "outputs/test")

    with patch(
        "sys.argv",
        ["train.py", "--stage", "stage2", "--config", "configs/stage2/stage2_qwen25_7b.yaml"],
    ):
        train.main()

    assert called["stage2"] is True
