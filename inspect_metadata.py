import argparse
from pathlib import Path
from pprint import pprint

import torch


DEFAULT_FOLDER = Path(r"pretrained\my_checkpoints")


def inspect_file(path: Path) -> None:
    print("=" * 80)
    print("FILE:", path.name)

    data = torch.load(path, map_location="cpu")
    print("TYPE:", type(data))

    if isinstance(data, dict):
        print("KEYS:", list(data.keys()))
        print("ITER:", data.get("iter"))
        print("FULL CONTENT:")
        pprint(data)
    else:
        print("RAW:")
        pprint(data)

    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Switti metadata .pt files.")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Optional metadata file path. If omitted, inspect all metadata*.pt files.",
    )
    parser.add_argument(
        "--folder",
        default=str(DEFAULT_FOLDER),
        help="Folder to scan when no file path is provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.path is not None:
        path = Path(args.path)
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        inspect_file(path)
        return

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Metadata folder not found: {folder}")

    paths = sorted(folder.glob("metadata*.pt"))
    if not paths:
        raise FileNotFoundError(f"No metadata*.pt files found in {folder}")

    for path in paths:
        inspect_file(path)


if __name__ == "__main__":
    main()
