import os
import torch

from models import SwittiPipeline


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


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")

    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    model_path = "./pretrained/Switti"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    default_prompt = "medieval stone wall texture, ancient masonry, archaeological site Eski-Kermen"
    prompt = ask_prompt(default_prompt)
    seed = ask_seed()

    print("Loading official Switti baseline...")
    pipe = SwittiPipeline.from_pretrained(
        model_path,
        device=device,
        torch_dtype=dtype,
        reso=512,
    )
    print("✅ Baseline pipeline loaded")

    print("Generating...")
    with torch.no_grad():
        images = pipe(
            [prompt],
            cfg=6.0,
            top_k=400,
            top_p=0.95,
            more_smooth=True,
            return_pil=True,
            smooth_start_si=2,
            turn_on_cfg_start_si=2,
            turn_off_cfg_start_si=11,
            last_scale_temp=0.1,
            seed=seed,
        )

    if images:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            results_dir,
            f"baseline_switti_seed{seed}_{timestamp}.png",
        )
        images[0].save(output_path)
        print(f"✅ Done! Saved as {output_path}")
    else:
        print("❌ No image generated")


if __name__ == "__main__":
    main()