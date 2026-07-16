#!/usr/bin/env python3
"""Run a prompt through the locally downloaded Qwen math model."""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "Qwen2.5-Math-7B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument("--prompt", help="Prompt text to send to Qwen")
    prompt_source.add_argument(
        "--prompt-file",
        type=Path,
        help="Read the prompt from a UTF-8 text file",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def get_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return args.prompt_file.expanduser().read_text(encoding="utf-8").strip()
    if args.prompt:
        return args.prompt.strip()
    return input("Enter your math prompt: ").strip()


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    if not (model_dir / "config.json").is_file():
        raise SystemExit(f"Model not found at: {model_dir}")

    prompt = get_prompt(args)
    if not prompt:
        raise SystemExit("The prompt cannot be empty.")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies. Install them with:\n"
            "  python3 -m pip install -r requirements.txt"
        ) from exc

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    print(f"Loading model from {model_dir} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=dtype,
    ).to(device)

    messages = [
        {"role": "system", "content": "You are a helpful mathematics assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    generation_args = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_args["temperature"] = args.temperature

    with torch.inference_mode():
        output_ids = model.generate(**generation_args)

    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print("\nQwen:\n" + answer)


if __name__ == "__main__":
    main()
