import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.color import lab2rgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.unet_colorizer import UNetColorizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("image", nargs="?", default="test.jpg")
    p.add_argument("--checkpoint", default="checkpoints/stage1_colorizer_latest.pth")
    p.add_argument("--output", default="output_stage1.jpg")
    p.add_argument(
        "--boost",
        type=float,
        default=1.6,
        help="AB multiplier. 1.6 = honest. 2.5 = vivid. 3.5 = over-saturated.",
    )
    p.add_argument("--ab-clip", type=float, default=95.0)
    p.add_argument("--side-by-side", action="store_true", help="Save greyscale + colourised side by side")
    return p.parse_args()


def load_model(ckpt_path):
    m = UNetColorizer(in_channels=1, out_channels=2)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in ckpt.get("model_state_dict", ckpt).items()}
    m.load_state_dict(sd, strict=False)
    m.eval()
    print(f"[OK] {ckpt_path}")
    return m


def semantic_correct(ab, l):
    h, w = l.shape
    out = ab.copy()
    lu8 = (l * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(lu8, (21, 21), 0).astype(np.float32) / 255
    tex = np.abs(l - blur)

    # Sky: bright + smooth + upper image
    sky = np.zeros((h, w), bool)
    sky[: int(h * 0.45), :] = True
    sky &= (l > 0.50) & (tex < 0.07)
    sky_needs_fix = sky & (ab[:, :, 1] < 0.15)
    b = 0.70
    out[sky_needs_fix, 0] = (1 - b) * ab[sky_needs_fix, 0] + b * (-0.04)
    out[sky_needs_fix, 1] = (1 - b) * ab[sky_needs_fix, 1] + b * (0.38)

    # Water: lower, smooth, mid-bright
    wat = np.zeros((h, w), bool)
    wat[int(h * 0.4) :, :] = True
    wat &= (l > 0.28) & (l < 0.72) & (tex < 0.025)
    wat_needs_fix = wat & (ab[:, :, 1] < 0.10)
    b = 0.55
    out[wat_needs_fix, 0] = (1 - b) * ab[wat_needs_fix, 0] + b * (-0.04)
    out[wat_needs_fix, 1] = (1 - b) * ab[wat_needs_fix, 1] + b * (0.22)

    # Vegetation: model predicts slight green, reinforce it.
    veg = (l > 0.15) & (l < 0.58) & (ab[:, :, 0] < -0.025)
    b = 0.40
    out[veg, 0] = (1 - b) * ab[veg, 0] + b * (-0.18)
    out[veg, 1] = (1 - b) * ab[veg, 1] + b * (-0.07)

    # Shadow desaturation.
    out[l < 0.08, :] *= 0.25
    return out


def bilateral_filter(ab128, guide):
    del guide

    def filt(ch):
        u8 = np.clip((ch + 128) / 256 * 255, 0, 255).astype(np.uint8)
        f = cv2.bilateralFilter(u8, d=9, sigmaColor=18, sigmaSpace=18)
        return f.astype(np.float32) / 255 * 256 - 128

    out = ab128.copy()
    out[:, :, 0] = filt(ab128[:, :, 0])
    out[:, :, 1] = filt(ab128[:, :, 1])
    return out


def quality_report(ab_norm):
    a, b = ab_norm[:, :, 0], ab_norm[:, :, 1]
    c = np.sqrt(a**2 + b**2)
    mc = float(c.mean())
    grade = "EXCELLENT" if mc > 0.22 else "GOOD" if mc > 0.14 else "FAIR" if mc > 0.07 else "POOR - keep training"
    print(f"  Mean chroma   : {mc:.3f}  [{grade}]")
    print(f"  Vivid pixels  : {float((c > 0.15).mean() * 100):.1f}%")
    print(f"  AB std        : A={float(np.std(a)):.3f}  B={float(np.std(b)):.3f}")
    return mc


def main():
    args = parse_args()
    model = load_model(args.checkpoint)

    gray_orig = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if gray_orig is None:
        raise FileNotFoundError(f"Cannot open: {args.image}")
    oh, ow = gray_orig.shape

    gray256 = cv2.resize(gray_orig, (256, 256), interpolation=cv2.INTER_AREA)
    l = gray256.astype(np.float32) / 255.0
    t = torch.from_numpy(l).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        ab = model(t).squeeze().permute(1, 2, 0).numpy()

    print("\n-- Raw model output --")
    mc_raw = quality_report(ab)

    # Semantic correction (only fixes what model clearly got wrong).
    ab = semantic_correct(ab, l)

    # Scale to [-128,128].
    ab128 = np.clip(ab * 128.0 * args.boost, -args.ab_clip, args.ab_clip)

    # Bilateral filter reduces colour bleeding.
    ab128 = bilateral_filter(ab128, gray256)

    # Reconstruct LAB -> RGB.
    lab = np.zeros((256, 256, 3), np.float32)
    lab[:, :, 0] = l * 100.0
    lab[:, :, 1:] = ab128
    rgb = np.clip(lab2rgb(lab), 0, 1)
    rgb = (rgb * 255).astype(np.uint8)

    # Resize to original.
    rgb_out = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
    bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)

    if args.side_by_side:
        gray_rgb = cv2.cvtColor(cv2.resize(gray_orig, (ow, oh)), cv2.COLOR_GRAY2BGR)
        combined = np.concatenate([gray_rgb, bgr_out], axis=1)
        sbs_path = args.output.replace(".jpg", "_compare.jpg")
        cv2.imwrite(sbs_path, combined)
        print(f"[OK] Side-by-side: {sbs_path}")

    cv2.imwrite(args.output, bgr_out)
    print(f"[OK] Output: {args.output}  ({ow}x{oh})")

    if mc_raw < 0.10:
        print("\nWARN: MODEL NEEDS MORE TRAINING")
        print("   Mean chroma < 0.10 means colours are too weak to post-process cleanly.")
        print("   Run training for 40+ more epochs, then re-test.")
    elif mc_raw < 0.15:
        print("\nINFO: Getting there - try 20 more training epochs for better results.")
    else:
        print("\nOK: Model output is strong. Results should look good.")

    print("\nTips:")
    print(f"  Stronger colours : --boost {args.boost + 0.4:.1f}")
    print(f"  Weaker colours   : --boost {max(1.0, args.boost - 0.4):.1f}")
    print("  Side by side     : --side-by-side")


if __name__ == "__main__":
    main()
