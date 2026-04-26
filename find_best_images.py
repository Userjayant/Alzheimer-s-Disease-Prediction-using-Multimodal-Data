"""
find_best_images.py
===================
Standalone helper script — does NOT modify any existing project file.

Usage
-----
  python find_best_images.py
  python find_best_images.py --folder train/ModerateDemented --top 10 --sample 200
"""

import os
import sys
import shutil
import random
import argparse
import numpy as np
import cv2

# ── Suppress TF noise before any TF import ────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as kimage

# ── Import YOUR existing functions — nothing is modified ──────────────────────
from predict import predict_alzheimer
from gradcam import get_gradcam_heatmap


# ================================================================
#  CONFIGURATION — edit these values, touch nothing else
# ================================================================
CONFIG = {
    # Folder containing your MRI images (subfolders scanned automatically)
    "image_folder":   "train/ModerateDemented",

    # Where the best images will be copied
    "output_folder":  "best_samples",

    # How many top images to select
    "top_n":          10,

    # Randomly sample this many images instead of scanning all 10,000
    # Set to 9999999 to scan everything (will be very slow on CPU)
    "sample_size":    200,

    # Fixed clinical data used for scoring: [age, mmse, cdr]
    "clinical_data":  [75.0, 18.0, 2.0],

    # Grad-CAM blend intensity (0.3–0.8)
    "gradcam_alpha":  0.55,

    # Score weights (must sum to 1.0)
    "weight_confidence": 0.5,
    "weight_heatmap":    0.5,

    # Skip images below this confidence %
    "min_confidence": 50.0,

    # Supported image extensions
    "extensions": (".jpg", ".jpeg", ".png", ".bmp"),
}


# ================================================================
#  MODEL SINGLETON — load once, reuse everywhere
# ================================================================
_model_instance = None

def _get_model():
    global _model_instance
    if _model_instance is None:
        print("  Loading model...")
        from tensorflow.keras.models import load_model as _load
        _model_instance = _load("alzheimer_multimodal_model.h5")
        print("  Model loaded.\n")
    return _model_instance


# ================================================================
#  SCORE FUNCTIONS
# ================================================================
def compute_heatmap_strength(heatmap: np.ndarray) -> float:
    """
    Score how much red/yellow the heatmap has (vs blue).
    Returns float in [0, 1]. Higher = stronger activation = better for demo.
    """
    if heatmap is None or heatmap.size == 0:
        return 0.0

    flat             = heatmap.flatten()
    mean_activation  = float(np.mean(flat))
    top_threshold    = np.percentile(flat, 75)
    top_pixels       = flat[flat >= top_threshold]
    peak_activation  = float(np.mean(top_pixels)) if len(top_pixels) > 0 else 0.0

    strength = 0.4 * mean_activation + 0.6 * peak_activation
    return float(np.clip(strength, 0.0, 1.0))


def score_image(confidence_pct: float, heatmap_strength: float) -> float:
    """Combine confidence + heatmap strength into one demo score."""
    conf_norm = confidence_pct / 100.0
    score = (
        CONFIG["weight_confidence"] * conf_norm +
        CONFIG["weight_heatmap"]    * heatmap_strength
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)


# ================================================================
#  MAIN SCANNER
# ================================================================
def find_best_images(image_folder: str, top_n: int, sample_size: int):

    # ── Validate folder ───────────────────────────────────────────────────
    if not os.path.isdir(image_folder):
        print(f"\n[ERROR] Folder not found: '{image_folder}'")
        print("        Edit CONFIG['image_folder'] in this script.")
        sys.exit(1)

    # ── Collect all image paths recursively ───────────────────────────────
    all_files = []
    for root, dirs, files in os.walk(image_folder):
        for f in files:
            if f.lower().endswith(CONFIG["extensions"]):
                all_files.append(os.path.join(root, f))

    if not all_files:
        print(f"\n[ERROR] No images found in '{image_folder}'")
        print(f"        Supported formats: {CONFIG['extensions']}")
        sys.exit(1)

    total_found = len(all_files)

    # ── Random sampling to keep runtime manageable ────────────────────────
    if sample_size < total_found:
        random.shuffle(all_files)
        all_files = all_files[:sample_size]
        print(f"\n  Sampling {sample_size} random images from {total_found:,} total.")
    else:
        print(f"\n  Scanning all {total_found:,} images (no sampling).")

    total = len(all_files)

    # Estimate time
    est_min = round(total * 4 / 60)
    print(f"  Estimated time : ~{est_min} minutes on CPU")

    print(f"\n{'='*60}")
    print(f"  Grad-CAM Demo Image Finder")
    print(f"{'='*60}")
    print(f"  Folder   : {image_folder}")
    print(f"  Scanning : {total} images")
    print(f"  Top N    : {top_n}")
    print(f"  Min conf : {CONFIG['min_confidence']}%")
    print(f"{'='*60}\n")

    clinical_data = np.array(CONFIG["clinical_data"], dtype=np.float32)
    results       = []
    skipped       = 0
    errors        = 0

    for idx, image_path in enumerate(all_files, start=1):

        filename = os.path.basename(image_path)

        # Progress every 10 images
        if idx % 10 == 0 or idx == 1:
            print(f"  [{idx:>4}/{total}]  Scanning...  "
                  f"({len(results)} candidates found so far)")

        try:
            # ── Prediction ────────────────────────────────────────────────
            result = predict_alzheimer(
                image_path        = image_path,
                clinical_data     = clinical_data.copy(),
                gradcam_intensity = CONFIG["gradcam_alpha"],
            )

            confidence = result.get("confidence", 0)

            if confidence < CONFIG["min_confidence"]:
                skipped += 1
                continue

            # ── Heatmap scoring ───────────────────────────────────────────
            img       = kimage.load_img(image_path, target_size=(128, 128))
            img_array = kimage.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            clin      = clinical_data[np.newaxis, :] if clinical_data.ndim == 1 else clinical_data

            heatmap = get_gradcam_heatmap(
                model                = _get_model(),
                img_array            = img_array,
                clinical_data        = clin,
                last_conv_layer_name = "auto",
            )

            heatmap_strength = compute_heatmap_strength(heatmap)
            demo_score       = score_image(confidence, heatmap_strength)

            results.append({
                "filename":         filename,
                "filepath":         image_path,
                "predicted_class":  result.get("class", "Unknown"),
                "confidence":       confidence,
                "heatmap_strength": round(heatmap_strength * 100, 2),
                "score":            demo_score,
                "gradcam_path":     result.get("gradcam_url"),
            })

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [SKIP] {filename}: {e}")

    # ── Sort and return top N ─────────────────────────────────────────────
    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n  Done.")
    print(f"  Scanned   : {total} images")
    print(f"  Skipped   : {skipped}  (confidence < {CONFIG['min_confidence']}%)")
    print(f"  Errors    : {errors}")
    print(f"  Qualified : {len(results)} images above threshold")

    return results[:top_n]


# ================================================================
#  DISPLAY RESULTS
# ================================================================
def print_results(top_results: list):

    if not top_results:
        print("\n  No qualifying images found.")
        print("  Try lowering CONFIG['min_confidence'] or increase --sample size.")
        return

    print(f"\n{'='*80}")
    print(f"  TOP {len(top_results)} IMAGES FOR DEMO")
    print(f"{'='*80}")
    print(f"  {'#':<4} {'Filename':<35} {'Confidence':>10} {'Heatmap':>10} {'Score':>8}")
    print(f"  {'-'*76}")

    for rank, r in enumerate(top_results, start=1):
        name = r["filename"]
        if len(name) > 33:
            name = name[:30] + "..."

        strength = r["heatmap_strength"]
        if strength >= 70:
            indicator = "Strong"
        elif strength >= 45:
            indicator = "Medium"
        else:
            indicator = "Weak  "

        print(
            f"  {rank:<4} {name:<35} "
            f"{r['confidence']:>9.1f}% "
            f"{strength:>7.1f}% {indicator}  "
            f"  {r['score']:.4f}"
        )

    print(f"{'='*80}\n")


# ================================================================
#  SAVE TOP IMAGES
# ================================================================
def save_top_images(top_results: list, output_folder: str):

    os.makedirs(output_folder, exist_ok=True)
    saved = 0
    print(f"  Saving top images to '{output_folder}/'...\n")

    for rank, r in enumerate(top_results, start=1):
        try:
            new_name = f"rank{rank:02d}_conf{r['confidence']:.0f}_{r['filename']}"
            shutil.copy2(r["filepath"], os.path.join(output_folder, new_name))

            if r["gradcam_path"] and os.path.exists(r["gradcam_path"]):
                gc_name = f"rank{rank:02d}_GRADCAM_{r['filename']}"
                shutil.copy2(r["gradcam_path"], os.path.join(output_folder, gc_name))

            print(f"  [{rank:>2}] Saved: {new_name}")
            saved += 1

        except Exception as e:
            print(f"  [{rank:>2}] Failed: {r['filename']} — {e}")

    print(f"\n  {saved} image(s) saved to '{output_folder}/'")


# ================================================================
#  ENTRY POINT
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Find best MRI images for Grad-CAM demo."
    )
    parser.add_argument(
        "--folder", type=str,
        default=CONFIG["image_folder"],
        help=f"MRI image folder (default: {CONFIG['image_folder']})"
    )
    parser.add_argument(
        "--top", type=int,
        default=CONFIG["top_n"],
        help=f"Number of top images to select (default: {CONFIG['top_n']})"
    )
    parser.add_argument(
        "--sample", type=int,
        default=CONFIG["sample_size"],
        help=f"Randomly sample N images for speed (default: {CONFIG['sample_size']})"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print results only, do not copy images to best_samples/"
    )
    args = parser.parse_args()

    _get_model()

    top_results = find_best_images(
        image_folder = args.folder,
        top_n        = args.top,
        sample_size  = args.sample,
    )

    print_results(top_results)

    if not args.no_save:
        save_top_images(top_results, CONFIG["output_folder"])
    else:
        print("  (--no-save flag: images not copied)")


if __name__ == "__main__":
    main()