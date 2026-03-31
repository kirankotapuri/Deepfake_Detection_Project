"""
Monitor cross_eval and subsequent pipeline progress.

Run in a separate terminal:
    python monitor_progress.py
"""
import time
from pathlib import Path
import os

CROSS_CKPTS = {
    "celebdf_resnet.pt", "celebdf_clip.pt", "celebdf_dinov2.pt",
    "uadfv_resnet.pt",   "uadfv_clip.pt",   "uadfv_dinov2.pt",
}
EXPECTED_CMS = [
    "cm_CelebDF_DFDC_resnet.png",  "cm_CelebDF_DFDC_clip.png",  "cm_CelebDF_DFDC_dinov2.png",
    "cm_CelebDF_UADFV_resnet.png", "cm_CelebDF_UADFV_clip.png", "cm_CelebDF_UADFV_dinov2.png",
    "cm_UADFV_CelebDF_resnet.png", "cm_UADFV_CelebDF_clip.png", "cm_UADFV_CelebDF_dinov2.png",
    "cm_UADFV_DFDC_resnet.png",    "cm_UADFV_DFDC_clip.png",    "cm_UADFV_DFDC_dinov2.png",
]

def check_state():
    ckpt_dir  = Path("checkpoints/cross")
    result_dir = Path("results")

    ckpts_done = [f.name for f in ckpt_dir.glob("*.pt")] if ckpt_dir.exists() else []
    cms_done   = sorted(f.name for f in result_dir.glob("cm_*.png")) if result_dir.exists() else []
    csv_exists = (result_dir / "cross_dataset_results.csv").exists()

    print(f"\n{'='*55}")
    print(f"  Progress report — {time.strftime('%H:%M:%S')}")
    print(f"{'='*55}")

    print(f"\n  Checkpoints ({len(ckpts_done)}/6):")
    for c in CROSS_CKPTS:
        status = "✔" if c in ckpts_done else "·"
        print(f"    [{status}] {c}")

    print(f"\n  Confusion matrices ({len(cms_done)}/12):")
    for c in EXPECTED_CMS:
        status = "✔" if c in cms_done else "·"
        print(f"    [{status}] {c}")

    print(f"\n  cross_dataset_results.csv : {'✔ DONE' if csv_exists else '· pending'}")

    if csv_exists:
        print("\n" + (result_dir / "cross_dataset_results.csv").read_text())

    done = csv_exists and all(c in ckpts_done for c in CROSS_CKPTS)
    return done


if __name__ == "__main__":
    print("Monitoring pipeline progress (Ctrl+C to stop)...")
    while True:
        done = check_state()
        if done:
            print("\nAll done! Run next steps:")
            print("  python main.py --mode explain --backbone resnet --gradcam --tsne")
            break
        time.sleep(60)
