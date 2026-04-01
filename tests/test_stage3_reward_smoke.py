from src.cope.stage3.simulator.plan_simulator import PlanSimulator


def test_stage3_reward_pipeline_smoke():
    registry = {
        "tokens": {
            "VQA_SMALL_HIGH_HIGH_HIGH_HIGH": {
                "resources": {
                    "cpu_core": 1,
                    "cpu_mem_gb": 1,
                    "gpu_sm": 1,
                    "gpu_mem_gb": 1,
                }
            }
        }
    }
    simulator = PlanSimulator(registry)
    plan = '<REF_0> = <EXEC> <VQA_SMALL_HIGH_HIGH_HIGH_HIGH>("img", "q")\n<FINISH> <REF_0>'
    state = {"cpu_core": 8, "cpu_memory": 16, "gpu_sm": 80, "gpu_memory": 16}
    result = simulator.evaluate(plan, state)
    assert isinstance(result.total, float)
