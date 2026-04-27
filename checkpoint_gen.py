import argparse
import os
import re
from datetime import datetime
from pathlib import Path

import torch

from models import SwittiPipeline


DEFAULT_PROMPT = (
    "medieval stone wall texture, ancient masonry, archaeological site Eski-Kermen"
)
DEFAULT_CHECKPOINT_DIR = "./pretrained/my_checkpoints"
DEFAULT_MODEL_PATH = "./pretrained/Switti"
RESULTS_DIR = "./results"


def ask_prompt(default_prompt: str) -> str:
    prompt_input = input("Enter prompt (press Enter to use default): ").strip()
    if prompt_input == "":
        print(f"Using default prompt: {default_prompt}")
        return default_prompt
    print(f"Using custom prompt: {prompt_input}")
    return prompt_input


def ask_seed() -> int:
    seed_input = input("Enter seed (press Enter for random): ").strip()

    if seed_input == "":
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        print(f"Using random seed: {seed}")
        return seed

    try:
        seed = int(seed_input)
        print(f"Using manual seed: {seed}")
        return seed
    except ValueError:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        print(f"Invalid seed input, using random seed: {seed}")
        return seed


def parse_checkpoint_number(path: Path, patterns: tuple[re.Pattern, ...]) -> int | None:
    for pattern in patterns:
        match = pattern.fullmatch(path.name)
        if match:
            return int(match.group(1))
    return None


def discover_checkpoints(checkpoint_dir: Path) -> list[dict]:
    metadata_pattern = re.compile(r"metadata_(\d+)\.pt")
    model_patterns = (
        re.compile(r"model_state_dict_(\d+)\.pt"),
        re.compile(r"model_(\d+)_state_dict\.pt"),
    )

    metadata_by_iter: dict[int, Path] = {}
    model_by_iter: dict[int, Path] = {}

    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue

        metadata_iter = parse_checkpoint_number(path, (metadata_pattern,))
        if metadata_iter is not None:
            metadata_by_iter[metadata_iter] = path
            continue

        model_iter = parse_checkpoint_number(path, model_patterns)
        if model_iter is not None:
            model_by_iter[model_iter] = path

    checkpoints = []
    for iteration in sorted(metadata_by_iter.keys() & model_by_iter.keys()):
        checkpoints.append(
            {
                "iteration": iteration,
                "metadata_path": metadata_by_iter[iteration],
                "model_path": model_by_iter[iteration],
            }
        )
    return checkpoints


def print_checkpoints(checkpoints: list[dict]) -> None:
    print("Available checkpoints:")
    for index, checkpoint in enumerate(checkpoints, start=1):
        iteration = checkpoint["iteration"]
        metadata_name = checkpoint["metadata_path"].name
        model_name = checkpoint["model_path"].name
        print(f"  {index}. iter {iteration}: {metadata_name} + {model_name}")


def choose_checkpoint(checkpoints: list[dict], requested_iteration: int | None) -> dict:
    if requested_iteration is not None:
        for checkpoint in checkpoints:
            if checkpoint["iteration"] == requested_iteration:
                print(f"Using checkpoint iteration: {requested_iteration}")
                return checkpoint
        available = ", ".join(str(item["iteration"]) for item in checkpoints)
        raise ValueError(
            f"Checkpoint {requested_iteration} not found. Available: {available}"
        )

    while True:
        choice = input("Select checkpoint by list number or iteration: ").strip()
        try:
            value = int(choice)
        except ValueError:
            print("Please enter a number from the list or checkpoint iteration.")
            continue

        if 1 <= value <= len(checkpoints):
            checkpoint = checkpoints[value - 1]
            print(f"Using checkpoint iteration: {checkpoint['iteration']}")
            return checkpoint

        for checkpoint in checkpoints:
            if checkpoint["iteration"] == value:
                print(f"Using checkpoint iteration: {checkpoint['iteration']}")
                return checkpoint

        print("Checkpoint not found. Try again.")


def load_checkpoint_into_pipeline(pipe: SwittiPipeline, checkpoint: dict) -> dict:
    model_state_dict = torch.load(checkpoint["model_path"], map_location="cpu")
    metadata = torch.load(checkpoint["metadata_path"], map_location="cpu")
    pipe.switti.load_state_dict(model_state_dict, strict=True)
    pipe.switti.eval()
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an image from a selected local Switti checkpoint."
    )
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=6.0)
    parser.add_argument("--top_k", type=int, default=400)
    parser.add_argument("--top_p", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint folder not found: {checkpoint_dir}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    checkpoints = discover_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError(
            "No paired checkpoints found. Expected metadata_{num}.pt plus "
            "model_state_dict_{num}.pt or model_{num}_state_dict.pt."
        )

    print_checkpoints(checkpoints)
    checkpoint = choose_checkpoint(checkpoints, args.checkpoint)

    prompt = args.prompt if args.prompt is not None else ask_prompt(DEFAULT_PROMPT)
    seed = args.seed if args.seed is not None else ask_seed()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading Switti pipeline...")
    pipe = SwittiPipeline.from_pretrained(
        str(model_path),
        device=device,
        torch_dtype=dtype,
        reso=512,
    )

    print(f"Loading fine-tuned checkpoint: {checkpoint['model_path']}")
    metadata = load_checkpoint_into_pipeline(pipe, checkpoint)
    print(f"Loaded checkpoint metadata iter: {metadata.get('iter', 'unknown')}")

    print("Generating...")
    with torch.no_grad():
        images = pipe(
            [prompt],
            cfg=args.cfg,
            top_k=args.top_k,
            top_p=args.top_p,
            more_smooth=True,
            return_pil=True,
            smooth_start_si=2,
            turn_on_cfg_start_si=2,
            turn_off_cfg_start_si=11,
            last_scale_temp=0.1,
            seed=seed,
        )

    if not images:
        print("No image generated")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        RESULTS_DIR,
        f"checkpoint_{checkpoint['iteration']}_seed{seed}_{timestamp}.png",
    )
    images[0].save(output_path)
    print(f"Done. Saved as {output_path}")


if __name__ == "__main__":
    main()
