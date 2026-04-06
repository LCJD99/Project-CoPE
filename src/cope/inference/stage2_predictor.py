"""
Stage 2 Planning Inference and Evaluation

This module provides inference capabilities for Stage 2 planning models.
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.cope.stage2.model import Stage2PlannerModel
from src.cope.inference.stage2_evaluator import Stage2Evaluator


def format_input_prompt(user_query: str, system_state: Dict[str, int]) -> str:
    """Format input prompt for Stage 2 model (must match training template)."""
    template = """[SYSTEM_STATE]
CPU_CORES: {cpu_core}
CPU_MEM: {cpu_memory} GB
GPU_SM: {gpu_sm} %
GPU_MEM: {gpu_memory} GB

[USER_QUERY]
{user_query}

[PLAN_START]
"""
    return template.format(
        cpu_core=system_state["cpu_core"],
        cpu_memory=system_state["cpu_memory"],
        gpu_sm=system_state["gpu_sm"],
        gpu_memory=system_state["gpu_memory"],
        user_query=user_query,
    )


def extract_plan_from_generation(generated_text: str, prompt: str) -> str:
    """Extract the generated plan from full generation output."""
    if generated_text.startswith(prompt):
        plan = generated_text[len(prompt) :].strip()
    elif "[PLAN_START]" in generated_text:
        plan = generated_text.split("[PLAN_START]")[-1].strip()
    else:
        plan = generated_text.strip()

    # Remove common EOS tokens
    for eos_token in ["<|endoftext|>", "<|im_end|>", "</s>"]:
        if eos_token in plan:
            plan = plan.split(eos_token)[0].strip()

    return plan


def load_stage2_model(
    checkpoint_path: str,
    device: str = "cuda",
    torch_dtype=torch.bfloat16,
    base_model_name: Optional[str] = None,
    stage1_checkpoint: Optional[str] = None,
) -> Tuple[Stage2PlannerModel, AutoTokenizer]:
    """Load trained Stage 2 planning model."""
    checkpoint_path = Path(checkpoint_path)

    print(f"\n{'=' * 60}")
    print(f"Loading Stage 2 Planning Model")
    print(f"{'=' * 60}")

    # Load config
    config_path = checkpoint_path.parent / "training_config.yaml"
    if not config_path.exists():
        config_path = checkpoint_path.parent.parent / "training_config.yaml"

    config: Dict[str, str] = {}
    if config_path.exists():
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if stage1_checkpoint is None:
            stage1_checkpoint = config.get("stage1_checkpoint")
        print(f"✓ Config loaded")
    elif stage1_checkpoint is None:
        raise FileNotFoundError(f"Config not found")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    print(f"✓ Tokenizer loaded (vocab: {len(tokenizer)})")

    # Load base model
    if base_model_name is None:
        base_model_name = config.get("base_model", "Qwen2.5-7B")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        base_model = base_model.to(device)
    print(f"✓ Base model loaded")

    # Load LoRA
    lora_path = checkpoint_path / "lora_weights"
    if lora_path.exists():
        base_model = PeftModel.from_pretrained(
            base_model, str(lora_path), is_trainable=False
        )
        print(f"✓ LoRA loaded")

    if not stage1_checkpoint:
        raise ValueError(
            "stage1_checkpoint is required to load Stage2 wrapper model. "
            "Provide it via training_config.yaml or infer config."
        )

    # Create Stage2 wrapper
    # Use checkpoint_path for stage2_embeddings_path (loaded from final_model/)
    model = Stage2PlannerModel(
        llm=base_model,
        tokenizer=tokenizer,
        stage1_collapsed_checkpoint=stage1_checkpoint,
        stage2_embeddings_path=str(checkpoint_path),
        device=device,
    )
    print(f"✓ Stage 2 model created")

    model.eval()
    print(f"✓ Model ready\n")

    return model, tokenizer


class Stage2Predictor:
    """Stage 2 planning predictor."""

    def __init__(
        self, model, tokenizer, device=None, max_new_tokens=512, generation_config=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = (
            next(model.parameters()).device if device is None else torch.device(device)
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.generation_config = dict(generation_config or {"do_sample": False})
        self.restrict_decode_to_new_and_tools = bool(
            self.generation_config.pop("restrict_decode_to_new_and_tools", True)
        )
        if hasattr(self.model, "set_decode_vocab_restriction"):
            self.model.set_decode_vocab_restriction(
                self.restrict_decode_to_new_and_tools
            )

    def predict(self, user_query: str, system_state: Dict[str, int], **kwargs) -> str:
        """Generate execution plan."""
        prompt = format_input_prompt(user_query, system_state)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        gen_config = {**self.generation_config, **kwargs}

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=self.max_new_tokens,
                **gen_config,
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return extract_plan_from_generation(generated_text, prompt)

    def predict_batch(
        self, examples: List[Dict], show_progress=True, **kwargs
    ) -> List[Dict]:
        """Batch prediction."""
        results = []
        iterator = (
            tqdm(examples, desc="Generating plans") if show_progress else examples
        )

        for example in iterator:
            try:
                prediction = self.predict(
                    example["user_query"], example["system_state"], **kwargs
                )
                result = {
                    "user_query": example["user_query"],
                    "system_state": example["system_state"],
                    "predicted_plan": prediction,
                    "success": True,
                }
                for key in ["task_id", "scenario_id", "ground_truth_plan"]:
                    if key in example:
                        result[key] = example[key]
            except Exception as e:
                result = {
                    "user_query": example["user_query"],
                    "system_state": example["system_state"],
                    "predicted_plan": f"[ERROR: {str(e)}]",
                    "success": False,
                    "error": str(e),
                }
            results.append(result)

        return results

    def evaluate_on_data(
        self, data_path: str, output_path: str, num_samples=None, **kwargs
    ) -> Dict:
        """Evaluate on test data."""
        print(f"\n{'=' * 60}")
        print(f"Stage 2 Planning Evaluation")
        print(f"{'=' * 60}")

        # Load data
        with open(data_path, "r") as f:
            data = json.load(f)

        # Flatten
        examples = []
        for task in data:
            for plan in task["plans"]:
                examples.append(
                    {
                        "task_id": task["task_id"],
                        "scenario_id": plan["scenario_id"],
                        "system_state": plan["SYSTEM_STATE"],
                        "user_query": plan["USER_QUESTION"],
                        "ground_truth_plan": plan["PLAN_START"],
                    }
                )

        if num_samples and num_samples < len(examples):
            import random

            random.seed(42)
            examples = random.sample(examples, num_samples)

        print(f"Total examples: {len(examples)}\n")

        # Predict
        predictions = self.predict_batch(examples, **kwargs)

        # Evaluate
        evaluator = Stage2Evaluator()
        eval_results = evaluator.evaluate_predictions(predictions)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "summary": eval_results["summary"],
            "per_scenario_stats": eval_results["per_scenario_stats"],
            "predictions": predictions,
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        # Print
        print(f"\n{'=' * 60}")
        print(f"Results")
        print(f"{'=' * 60}")
        print(f"Total: {eval_results['summary']['total_samples']}")
        print(f"Exact matches: {eval_results['summary']['exact_match_count']}")
        print(f"Exact match rate: {eval_results['summary']['exact_match_rate']:.2%}")

        if eval_results["per_scenario_stats"]:
            print(f"\nPer-scenario:")
            for sid, stats in sorted(eval_results["per_scenario_stats"].items()):
                print(
                    f"  Scenario {sid}: {stats['exact_match_rate']:.2%} ({stats['exact_match_count']}/{stats['total']})"
                )

        print(f"\nSaved to: {output_path}\n")

        return output_data


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="evaluate", choices=["single", "evaluate"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--user_query")
    parser.add_argument("--cpu_core", type=int, default=8)
    parser.add_argument("--cpu_memory", type=int, default=16)
    parser.add_argument("--gpu_sm", type=int, default=50)
    parser.add_argument("--gpu_memory", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    model, tokenizer = load_stage2_model(args.checkpoint, device=args.device)
    predictor = Stage2Predictor(
        model, tokenizer, device=args.device, max_new_tokens=args.max_new_tokens
    )

    if args.mode == "single":
        if not args.user_query:
            parser.error("--user_query required")
        state = {
            "cpu_core": args.cpu_core,
            "cpu_memory": args.cpu_memory,
            "gpu_sm": args.gpu_sm,
            "gpu_memory": args.gpu_memory,
        }
        plan = predictor.predict(args.user_query, state)
        print(f"\nGenerated Plan:\n{plan}\n")
    else:
        if not args.data or not args.output:
            parser.error("--data and --output required")
        predictor.evaluate_on_data(args.data, args.output, args.num_samples)


if __name__ == "__main__":
    main()
