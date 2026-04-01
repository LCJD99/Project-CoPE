import infer


def test_infer_cli_accepts_stage2_and_stage3():
    args2 = infer.parse_args(
        ["--stage", "stage2", "--config", "a.yaml", "--input", "in.json", "--output", "out.json"]
    )
    args3 = infer.parse_args(
        ["--stage", "stage3", "--config", "a.yaml", "--input", "in.json", "--output", "out.json"]
    )
    assert args2.stage == "stage2"
    assert args3.stage == "stage3"
