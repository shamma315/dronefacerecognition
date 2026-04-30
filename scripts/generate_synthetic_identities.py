"""Generate 30 synthetic identities × 40 images = 1,200 images via Stable Diffusion 1.5.

Idempotent: skips images that already exist on disk, so re-running the script picks up
where it left off. Saves to /home/buthaina.almulla/Documents/CV7502/synthetic_identities/identity_<NNN>/img_<NNN>.jpg.
"""
import os, sys, time, argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def identity_descriptor(idx: int) -> str:
    """Stable demographic+appearance descriptor for identity idx (0..29).

    30 unique combos = 3 ages × 2 genders × 5 ethnicities. Hair varies on a
    cycle so visually adjacent identities differ further.
    """
    AGES = ["in their 20s", "middle-aged", "elderly"]
    GENDERS = ["man", "woman"]
    ETHNICITIES = ["European", "East Asian", "South Asian", "African", "Latino"]
    HAIR = [
        "short black hair",
        "long brown hair",
        "short blonde hair",
        "curly black hair",
        "straight gray hair",
    ]
    age = AGES[idx % 3]
    gender = GENDERS[(idx // 3) % 2]
    eth = ETHNICITIES[(idx // 6) % 5]
    hair = HAIR[idx % 5]
    return f"a {eth} {gender} {age} with {hair}"


def variation_prompt(j: int) -> str:
    """40 unique scene/style descriptors covering UAV-like + low-quality conditions."""
    CAMERA = [
        "aerial drone photograph from above",
        "low-angle surveillance camera footage",
        "outdoor security camera shot",
        "handheld camera at street level",
    ]
    LIGHTING = [
        "harsh sunlight",
        "overcast cloudy day",
        "soft dusk lighting",
        "low resolution CCTV quality",
        "slight motion blur",
    ]
    ANGLE = ["frontal view", "oblique three-quarter angle"]
    cam = CAMERA[j % 4]
    light = LIGHTING[(j // 4) % 5]
    ang = ANGLE[(j // 20) % 2]
    return f"{cam}, {light}, {ang}"


NEGATIVE = (
    "deformed, ugly, blurry, low quality, cartoon, illustration, painting, drawing, "
    "anime, bad anatomy, multiple faces, watermark, text"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="/home/buthaina.almulla/Documents/CV7502/synthetic_identities")
    ap.add_argument("--n-identities", type=int, default=30)
    ap.add_argument("--n-per-identity", type=int, default=40)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} (fp16)…", flush=True)
    t_load = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    print(f"loaded in {time.time() - t_load:.1f}s", flush=True)

    total = args.n_identities * args.n_per_identity
    done = sum(
        1
        for i in range(args.n_identities)
        for j in range(args.n_per_identity)
        if (out_root / f"identity_{i+1:03d}" / f"img_{j+1:03d}.jpg").exists()
    )
    print(f"target: {total} images | already done: {done}", flush=True)

    t0 = time.time()
    new_done = 0
    for i in range(args.n_identities):
        id_dir = out_root / f"identity_{i+1:03d}"
        id_dir.mkdir(parents=True, exist_ok=True)
        descriptor = identity_descriptor(i)
        for j in range(args.n_per_identity):
            out_path = id_dir / f"img_{j+1:03d}.jpg"
            if out_path.exists():
                continue
            prompt = (
                f"a photograph of {descriptor}, {variation_prompt(j)}, "
                f"photorealistic, sharp facial features, real person"
            )
            seed = i * 10000 + j
            gen = torch.Generator(device="cuda").manual_seed(seed)
            with torch.inference_mode():
                out = pipe(
                    prompt,
                    negative_prompt=NEGATIVE,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    generator=gen,
                )
            out.images[0].save(out_path, quality=90)
            done += 1
            new_done += 1
            if new_done % 25 == 0:
                elapsed = time.time() - t0
                rate = new_done / max(1, elapsed)
                eta = (total - done) / max(1e-3, rate)
                print(
                    f"[{done:4d}/{total}] id={i+1:02d} img={j+1:02d} | "
                    f"rate={rate:.2f} img/s | eta={eta/60:.1f}min",
                    flush=True,
                )

    print(f"done | total wall: {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
