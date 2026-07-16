#!/usr/bin/env python3
"""Select samples solved on simple perturbations but missed on hard ones."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
MATH_PERTURB_DIR = ROOT / "MATH-Perturb"
DEFAULT_SIMPLE_RESULTS = ROOT / "results" / "math_perturb_simple_qwen.jsonl"
DEFAULT_HARD_RESULTS = ROOT / "results" / "math_perturb_hard_qwen.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "interesting_samples_qwen.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simple-results", type=Path, default=DEFAULT_SIMPLE_RESULTS)
    parser.add_argument("--hard-results", type=Path, default=DEFAULT_HARD_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--match-key",
        choices=("row_index", "problem_id"),
        default="row_index",
        help="Field used to pair simple and hard results.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write only ids, answers, and correctness flags instead of full responses.",
    )
    return parser.parse_args()


def load_jsonl(path: Path, match_key: str) -> dict[Any, dict[str, Any]]:
    rows = {}
    with path.expanduser().open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if match_key not in row:
                raise SystemExit(f"Missing {match_key!r} in {path}:{line_number}")
            key = row[match_key]
            if key in rows:
                raise SystemExit(f"Duplicate {match_key}={key!r} in {path}")
            rows[key] = normalize_result(row)
    return rows


def normalize_result(row: dict[str, Any]) -> dict[str, Any]:
    """Support result files written before key names were normalized."""
    normalized = dict(row)
    if "ground_truth_answer" not in normalized and "answer" in normalized:
        normalized["ground_truth_answer"] = normalized["answer"]
    if "model_name" not in normalized and "model" in normalized:
        normalized["model_name"] = normalized["model"]
    return normalized


def get_answer_checker():
    sys.path.insert(0, str(MATH_PERTURB_DIR))
    try:
        from evaluate import answer_check
    except ImportError as exc:
        raise SystemExit(
            "Could not import the MATH-Perturb evaluator. "
            "Install its dependencies, then try again."
        ) from exc
    return answer_check


def is_correct(row: dict[str, Any], answer_check) -> bool:
    required_keys = ("problem", "model_response", "ground_truth_answer")
    missing = [key for key in required_keys if key not in row]
    if missing:
        raise SystemExit(
            f"Result for row_index={row.get('row_index')} is missing: {', '.join(missing)}"
        )
    return bool(
        answer_check(
            row["problem"],
            row["model_response"],
            row["ground_truth_answer"],
            dataset_type="perturb",
        )
    )


def build_output_row(
    key: Any,
    simple: dict[str, Any],
    hard: dict[str, Any],
    simple_correct: bool,
    hard_correct: bool,
    compact: bool,
) -> dict[str, Any]:
    base = {
        "match_key": key,
        "row_index": simple.get("row_index"),
        "problem_id": simple.get("problem_id"),
        "simple_is_correct": simple_correct,
        "hard_is_correct": hard_correct,
        "simple_ground_truth_answer": simple.get("ground_truth_answer"),
        "hard_ground_truth_answer": hard.get("ground_truth_answer"),
    }
    if compact:
        return base
    return {
        **base,
        "simple_problem": simple.get("problem"),
        "hard_problem": hard.get("problem"),
        "simple_model_response": simple.get("model_response"),
        "hard_model_response": hard.get("model_response"),
        "simple_metadata": simple.get("metadata", {}),
        "hard_metadata": hard.get("metadata", {}),
    }


def main() -> None:
    args = parse_args()
    simple_results = load_jsonl(args.simple_results, args.match_key)
    hard_results = load_jsonl(args.hard_results, args.match_key)
    answer_check = get_answer_checker()

    paired_keys = sorted(set(simple_results) & set(hard_results))
    missing_hard = len(set(simple_results) - set(hard_results))
    missing_simple = len(set(hard_results) - set(simple_results))

    interesting = []
    simple_correct_count = 0
    hard_correct_count = 0
    for key in paired_keys:
        simple = simple_results[key]
        hard = hard_results[key]
        simple_correct = is_correct(simple, answer_check)
        hard_correct = is_correct(hard, answer_check)
        simple_correct_count += int(simple_correct)
        hard_correct_count += int(hard_correct)
        if simple_correct and not hard_correct:
            interesting.append(
                build_output_row(
                    key,
                    simple,
                    hard,
                    simple_correct,
                    hard_correct,
                    compact=args.compact,
                )
            )

    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with args.output.expanduser().open("w", encoding="utf-8") as output_file:
        for row in interesting:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Compared paired samples: {len(paired_keys)}")
    print(f"Simple correct: {simple_correct_count}")
    print(f"Hard correct: {hard_correct_count}")
    print(f"Interesting samples: {len(interesting)}")
    print(f"Missing hard results for simple rows: {missing_hard}")
    print(f"Missing simple results for hard rows: {missing_simple}")
    print(f"Saved to: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
