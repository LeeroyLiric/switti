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


def get_prompt(prompt_arg: str | None, default_prompt: str, use_default: bool) -> str:
    if prompt_arg is not None or use_default:
        prompt = "" if prompt_arg is None else prompt_arg.strip()
        if prompt == "":
            print(f"Using default prompt: {default_prompt}")
            return default_prompt
        print(f"Using custom prompt: {prompt}")
        return prompt

    return ask_prompt(default_prompt)


def get_seed(seed_arg: int | None) -> int:
    if seed_arg is not None:
        print(f"Using manual seed: {seed_arg}")
        return seed_arg

    return ask_seed()


def parse_checkpoint_number(path: Path, patterns: tuple[re.Pattern, ...]) -> int | None:
    for pattern in patterns:
        match = pattern.fullmatch(path.name)
        if match:
            return int(match.group(1))
    return None


def load_metadata_iteration(metadata_path: Path) -> int | None:
    metadata = torch.load(metadata_path, map_location="cpu")
    iteration = metadata.get("iter")
    return iteration if isinstance(iteration, int) else None


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
        metadata_path = metadata_by_iter[iteration]
        metadata_iteration = load_metadata_iteration(metadata_path)
        checkpoints.append(
            {
                "iteration": metadata_iteration if metadata_iteration is not None else iteration,
                "file_iteration": iteration,
                "metadata_iteration": metadata_iteration,
                "metadata_path": metadata_path,
                "model_path": model_by_iter[iteration],
            }
        )
    return checkpoints


def print_checkpoints(checkpoints: list[dict]) -> None:
    print("Available checkpoints:")
    for index, checkpoint in enumerate(checkpoints, start=1):
        iteration = checkpoint["iteration"]
        file_iteration = checkpoint["file_iteration"]
        metadata_name = checkpoint["metadata_path"].name
        model_name = checkpoint["model_path"].name
        print(f"  {index}. iter {iteration}: {metadata_name} + {model_name}")
        if checkpoint["metadata_iteration"] != file_iteration:
            print(
                f"     WARNING: filename iter={file_iteration}, metadata iter={checkpoint['metadata_iteration']}"
            )


def find_checkpoint(checkpoints: list[dict], choice: str | int) -> dict:
    try:
        value = int(str(choice).strip())
    except ValueError as exc:
        raise ValueError(f"Checkpoint not found: {choice}") from exc

    if 1 <= value <= len(checkpoints):
        return checkpoints[value - 1]

    for checkpoint in checkpoints:
        if checkpoint["iteration"] == value:
            return checkpoint

    available = ", ".join(str(item["iteration"]) for item in checkpoints)
    raise ValueError(
        f"Checkpoint not found: {choice}. "
        f"Use list number 1-{len(checkpoints)} or iteration: {available}"
    )


def choose_checkpoint(checkpoints: list[dict], requested_checkpoint: str | None) -> dict:
    if requested_checkpoint is not None:
        checkpoint = find_checkpoint(checkpoints, requested_checkpoint)
        print(f"Using checkpoint iteration: {checkpoint['iteration']}")
        return checkpoint

    while True:
        choice = input("Select checkpoint by list number or iteration: ").strip()
        try:
            int(choice)
        except ValueError:
            print("Please enter a number from the list or checkpoint iteration.")
            continue

        try:
            checkpoint = find_checkpoint(checkpoints, choice)
        except ValueError:
            print("Checkpoint not found. Try again.")
            continue

        print(f"Using checkpoint iteration: {checkpoint['iteration']}")
        return checkpoint


def load_checkpoint_into_pipeline(pipe: SwittiPipeline, checkpoint: dict) -> dict:
    model_state_dict = torch.load(checkpoint["model_path"], map_location="cpu")
    metadata = torch.load(checkpoint["metadata_path"], map_location="cpu")
    pipe.switti.load_state_dict(model_state_dict, strict=True)
    pipe.switti.eval()
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an image from a selected local Switti checkpoint.",
        epilog=(
            "Examples:\n"
            "  python checkpoint_gen.py\n"
            "  python checkpoint_gen.py --checkpoint 1 --seed 697\n"
            "  python checkpoint_gen.py --checkpoint 100 --seed 697\n"
            "  python checkpoint_gen.py --checkpoint 200 "
            '--prompt "medieval stone wall texture, ancient masonry" --seed 850\n'
            '  python checkpoint_gen.py --checkpoint 100 --prompt "" --seed 850'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint list number (1, 2, ...) or iteration number (100, 200, ...).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text. Use an empty string to use the default prompt without input.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Manual generation seed.")
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
    try:
        checkpoint = choose_checkpoint(checkpoints, args.checkpoint)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from None

    use_default_prompt = (
        args.prompt is None and args.checkpoint is not None and args.seed is not None
    )
    prompt = get_prompt(args.prompt, DEFAULT_PROMPT, use_default_prompt)
    seed = get_seed(args.seed)

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
    if metadata.get("iter") != checkpoint["file_iteration"]:
        print(
            "WARNING: checkpoint filename iteration does not match metadata iteration "
            f"({checkpoint['file_iteration']} != {metadata.get('iter')})"
        )

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
    result_iteration = metadata.get("iter", checkpoint["iteration"])
    output_path = os.path.join(
        RESULTS_DIR,
        f"checkpoint_{result_iteration}_seed{seed}_{timestamp}.png",
    )
    images[0].save(output_path)
    print(f"Done. Saved as {output_path}")


if __name__ == "__main__":
    main()
