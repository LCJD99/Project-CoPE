import json

from src.cope.stage2.dataset import Stage2PlanningDataset


class DummyTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __len__(self):
        return 100

    def encode(self, text, add_special_tokens=True):
        base = [2, 3] if add_special_tokens else []
        body = [min(99, (ord(ch) % 50) + 2) for ch in text[:32]]
        return base + body


def test_stage2_dataset_loads_minimal_sample(tmp_path):
    payload = [
        {
            "task_id": "t1",
            "plans": [
                {
                    "scenario_id": 1,
                    "SYSTEM_STATE": {
                        "cpu_core": 8,
                        "cpu_memory": 16,
                        "gpu_sm": 50,
                        "gpu_memory": 8,
                    },
                    "USER_QUESTION": "q",
                    "PLAN_START": "<REF_0> = <EXEC> <TOOL>()\n<FINISH> <REF_0>",
                }
            ],
        }
    ]
    data = tmp_path / "sample.json"
    data.write_text(json.dumps(payload), encoding="utf-8")

    ds = Stage2PlanningDataset(str(data), tokenizer=DummyTokenizer(), max_length=64)
    assert len(ds) == 1
    item = ds[0]
    assert item["input_ids"].shape[0] == 64
    assert item["labels"].shape[0] == 64
