"""IP-Adapter FaceID v2 generation for 30 synthetic identities × 40 images.

For each identity:
  1. Generate one anchor portrait (text-to-image) with strong portrait prompt.
  2. Run insightface buffalo_l on the anchor to extract a 512-d FaceID embedding.
  3. Use diffusers IP-Adapter FaceID conditioning on that embedding to generate the
     remaining 39 images with variation prompts. The face stays consistent across
     all 40 images of the identity because all generations are conditioned on the
     same anchor face embedding.

Idempotent: skips images that already exist on disk.
Saves to synthetic_identities/identity_<NNN>/img_<NNN>.jpg.
Per-identity anchor face embedding is cached to anchor_embed.npy.
"""
import os, sys, time, argparse
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis


# ----- prompt ladders ------------------------------------------------------
def identity_descriptor(idx: int) -> str:
    AGES = ["in their 20s", "middle-aged", "elderly"]
    GENDERS = ["man", "woman"]
    ETHNICITIES = ["European", "East Asian", "South Asian", "African", "Latino"]
    HAIR = [
        "short black hair", "long brown hair", "short blonde hair",
        "curly black hair", "straight gray hair",
    ]
    return (
        f"a {ETHNICITIES[(idx//6)%5]} {GENDERS[(idx//3)%2]} "
        f"{AGES[idx%3]} with {HAIR[idx%5]}"
    )


def variation_prompt(j: int) -> str:
    CAMERA = [
        "aerial drone photograph from above",
        "low-angle surveillance camera footage",
        "outdoor security camera shot",
        "handheld camera at street level",
    ]
    LIGHTING = [
        "harsh sunlight", "overcast cloudy day", "soft dusk lighting",
        "low resolution CCTV quality", "slight motion blur",
    ]
    ANGLE = ["frontal view", "oblique three-quarter angle"]
    return (
        f"{CAMERA[j%4]}, {LIGHTING[(j//4)%5]}, {ANGLE[(j//20)%2]}"
    )


ANCHOR_TEMPLATE = (
    "a high quality portrait photograph of {desc}, "
    "looking at camera, frontal view, sharp facial features, "
    "photorealistic, detailed face, studio lighting, head and shoulders shot"
)
VARIATION_TEMPLATE = (
    "a photograph of {desc}, {var}, photorealistic, sharp facial features, real person, visible face"
)
NEGATIVE = (
    "deformed, ugly, blurry, low quality, cartoon, illustration, painting, drawing, "
    "anime, bad anatomy, multiple faces, watermark, text, child, baby"
)


# ----- helpers -------------------------------------------------------------
def get_face_embed(face_app: FaceAnalysis, pil_image: Image.Image):
    """Return (512,) normalised embedding or None if no face is detected."""
    rgb = np.array(pil_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    faces = face_app.get(bgr)
    if len(faces) == 0:
        return None
    # Largest face by bbox area, in case of multi-face anchors
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    return faces[0].normed_embedding.astype(np.float32)


def generate_anchor(pipe, descriptor: str, base_seed: int, face_app):
    """Generate an anchor portrait whose face insightface can detect; return (PIL, embedding).

    Caller must pass a pipe that does NOT have IP-Adapter loaded — once IP-Adapter is
    loaded, the UNet will demand `ip_adapter_image_embeds` in every call.
    """
    prompt = ANCHOR_TEMPLATE.format(desc=descriptor)
    for retry in range(8):
        seed = base_seed + retry
        gen = torch.Generator(device="cuda").manual_seed(seed)
        with torch.inference_mode():
            img = pipe(
                prompt=prompt, negative_prompt=NEGATIVE,
                height=512, width=512,
                num_inference_steps=30, guidance_scale=7.5,
                generator=gen,
            ).images[0]
        emb = get_face_embed(face_app, img)
        if emb is not None:
            return img, emb, seed
    raise RuntimeError(f"could not produce an anchor with detectable face for: {descriptor}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="/home/buthaina.almulla/Documents/CV7502/synthetic_identities")
    ap.add_argument("--n-identities", type=int, default=30)
    ap.add_argument("--n-per-identity", type=int, default=40)
    ap.add_argument("--var-steps", type=int, default=20)
    ap.add_argument("--ip-scale", type=float, default=0.7)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # --- insightface buffalo_l for face embedding extraction (CPU is fine) ---
    print("loading insightface (buffalo_l)…", flush=True)
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    # =====================================================================
    # PHASE A — generate all 30 anchors with plain text-to-image.
    # IP-Adapter is NOT loaded yet, because once it is loaded the UNet expects
    # `ip_adapter_image_embeds` in every forward — which we don't have for anchors.
    # =====================================================================
    print(f"loading {args.model} (fp16) for anchor generation…", flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    # NOTE: do NOT call enable_attention_slicing() — it installs a SlicedAttnProcessor
    # that conflicts with IP-Adapter's attention-processor injection (TypeError on load).
    # 32GB VRAM is plenty for SD 1.5 fp16 at 512x512 without slicing.

    print(f"\n=== Phase A: generating {args.n_identities} anchors ===", flush=True)
    t_a = time.time()
    anchors_meta = []  # [(id_dir, descriptor, embedding_np, seed)]
    for i in range(args.n_identities):
        id_dir = out_root / f"identity_{i+1:03d}"
        id_dir.mkdir(parents=True, exist_ok=True)
        descriptor = identity_descriptor(i)
        anchor_path = id_dir / "anchor.jpg"
        embed_path = id_dir / "anchor_embed.npy"

        if anchor_path.exists() and embed_path.exists():
            faceid_embeds = np.load(embed_path)
            anchor_seed = -1  # cached
        else:
            anchor_img, faceid_embeds, anchor_seed = generate_anchor(
                pipe, descriptor, base_seed=i * 7919 + 1234, face_app=face_app,
            )
            anchor_img.save(anchor_path, quality=95)
            np.save(embed_path, faceid_embeds)
        anchors_meta.append((id_dir, descriptor, faceid_embeds, anchor_seed))
        if (i + 1) % 5 == 0 or i == args.n_identities - 1:
            print(
                f"  anchors {i+1}/{args.n_identities} done "
                f"[{time.time()-t_a:.1f}s]", flush=True,
            )
    print(f"phase A done in {(time.time()-t_a)/60:.1f}min", flush=True)

    # =====================================================================
    # PHASE B — load IP-Adapter FaceID and generate the 40 variations.
    # =====================================================================
    print("\n=== Phase B: loading IP-Adapter FaceID + generating variations ===", flush=True)
    print("downloading IP-Adapter FaceID weights…", flush=True)
    faceid_path = hf_hub_download(
        repo_id="h94/IP-Adapter-FaceID",
        filename="ip-adapter-faceid_sd15.bin",
    )
    pipe.load_ip_adapter(
        faceid_path.rsplit("/", 1)[0],
        subfolder=None,
        weight_name="ip-adapter-faceid_sd15.bin",
        image_encoder_folder=None,
    )
    pipe.set_ip_adapter_scale(args.ip_scale)
    print("IP-Adapter FaceID loaded, scale =", args.ip_scale, flush=True)

    total = args.n_identities * args.n_per_identity
    t_b = time.time()
    new_done = 0
    for i, (id_dir, descriptor, faceid_embeds, anchor_seed) in enumerate(anchors_meta):
        face_t = torch.from_numpy(faceid_embeds).to("cuda", dtype=torch.float16).view(1, 1, 512)
        neg_face_t = torch.zeros_like(face_t)
        for j in range(args.n_per_identity):
            out_path = id_dir / f"img_{j+1:03d}.jpg"
            if out_path.exists():
                continue
            prompt = VARIATION_TEMPLATE.format(desc=descriptor, var=variation_prompt(j))
            seed = i * 10000 + j
            gen = torch.Generator(device="cuda").manual_seed(seed)
            with torch.inference_mode():
                out = pipe(
                    prompt=prompt, negative_prompt=NEGATIVE,
                    height=512, width=512,
                    num_inference_steps=args.var_steps,
                    guidance_scale=args.guidance,
                    generator=gen,
                    ip_adapter_image_embeds=[torch.cat([neg_face_t, face_t], dim=0)],
                )
            out.images[0].save(out_path, quality=90)
            new_done += 1

            if new_done % 20 == 0:
                elapsed = time.time() - t_b
                rate = new_done / max(1, elapsed)
                done_total = sum(
                    1 for ii in range(args.n_identities)
                    for jj in range(args.n_per_identity)
                    if (out_root / f"identity_{ii+1:03d}" / f"img_{jj+1:03d}.jpg").exists()
                )
                eta = (total - done_total) / max(1e-3, rate)
                print(
                    f"[{done_total:4d}/{total}] id={i+1:02d} img={j+1:02d} "
                    f"rate={rate:.2f}/s eta={eta/60:.1f}min",
                    flush=True,
                )

    print(f"phase B done | {new_done} new images in {(time.time()-t_b)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
