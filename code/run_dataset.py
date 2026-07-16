#!/usr/bin/env python3
"""Run every MATH-Perturb question through the local Qwen model."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen2.5-Math-7B-Instruct"
DEFAULT_DATASETS = [
    ROOT / "MATH-Perturb" / "math_perturb" / "math_perturb_simple.jsonl",
    ROOT / "MATH-Perturb" / "math_perturb" / "math_perturb_hard.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        type=Path,
        help="JSONL dataset to process; repeat for multiple files (default: simple and hard)",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many questions per dataset (useful for testing)",
    )
    return parser.parse_args()


def load_completed(output_path: Path) -> set[int]:
    """Return row indices already saved so an interrupted run can resume."""
    completed = set()
    if not output_path.exists():
        return completed
    with output_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                completed.add(json.loads(line)["row_index"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise SystemExit(
                    f"Invalid result at {output_path}:{line_number}; "
                    "repair or remove that line before resuming."
                ) from exc
    return completed


def normalize_saved_result_keys(output_path: Path) -> None:
    """Make older saved result files use clear result key names."""
    if not output_path.exists():
        return

    normalized_lines = []
    changed = False
    with output_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                normalized_lines.append(line)
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid result at {output_path}:{line_number}; "
                    "repair or remove that line before resuming."
                ) from exc

            if "ground_truth_answer" not in result and "answer" in result:
                result["ground_truth_answer"] = result.pop("answer")
                changed = True
            if "model_name" not in result and "model" in result:
                result["model_name"] = result.pop("model")
                changed = True

            normalized_lines.append(json.dumps(result, ensure_ascii=False) + "\n")

    if changed:
        output_path.write_text("".join(normalized_lines), encoding="utf-8")
        print(f"Updated saved result keys in {output_path}")


def main() -> None:
    args = parse_args()
    datasets = args.dataset or DEFAULT_DATASETS
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"Model not found at: {model_dir}")
    for dataset in datasets:
        if not dataset.expanduser().is_file():
            raise SystemExit(f"Dataset not found: {dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from tqdm import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install them with:\n"
            "  python3 -m pip install -r requirements.txt"
        ) from exc

    if torch.backends.mps.is_available():
        device, dtype = torch.device("mps"), torch.float16
    elif torch.cuda.is_available():
        device, dtype = torch.device("cuda"), torch.float16
    else:
        device, dtype = torch.device("cpu"), torch.float32

    print(f"Loading Qwen on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    for dataset_path in datasets:
        dataset_path = dataset_path.expanduser().resolve()
        with dataset_path.open(encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]
        if args.limit is not None:
            rows = rows[: args.limit]

        output_path = output_dir / f"{dataset_path.stem}_qwen.jsonl"
        normalize_saved_result_keys(output_path)
        completed = load_completed(output_path)
        remaining = [(index, row) for index, row in enumerate(rows) if index not in completed]
        print(
            f"{dataset_path.name}: {len(rows)} total, "
            f"{len(completed)} already complete, {len(remaining)} remaining"
        )

        with output_path.open("a", encoding="utf-8") as output_file:
            for row_index, row in tqdm(remaining, desc=dataset_path.stem, unit="question"):
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "Solve the mathematics problem carefully. Show your reasoning and "
                            "put the final answer inside \\boxed{}."
                        ),
                    },
                    {"role": "user", "content": row["problem"]},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = generated[0, inputs["input_ids"].shape[1] :]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)

                metadata = {
                    key: value
                    for key, value in row.items()
                    if key not in {"problem", "answer"}
                }
                result = {
                    "row_index": row_index,
                    "problem_id": row.get("problem_id"),
                    "problem": row["problem"],
                    "ground_truth_answer": row["answer"],
                    "model_name": "Qwen/Qwen2.5-Math-7B-Instruct",
                    "model_response": response,
                    "metadata": metadata,
                }
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                output_file.flush()

        print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
