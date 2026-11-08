#!/usr/bin/env python3
"""Unified Kaggle execution worker for automated SAR-guided RSID experiments.

Handles repository synchronization, dataset discovery, training execution,
full test evaluation, and negative-sample (clear-pixel preservation) assessment.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def find_mrshaze_root(candidates=None):
    """Dynamically discover MRSHaze root directory under /kaggle/input."""
    if candidates is None:
        candidates = [
            Path("/kaggle/input/datasets/markosumire/mrshaze/MRSHaze"),
            Path("/kaggle/input/datasets/markosumire/mrshaze"),
            Path("/kaggle/input/mrshaze/MRSHaze"),
            Path("/kaggle/input/mrshaze"),
            Path("/kaggle/input/MRSHaze"),
        ]
    for p in candidates:
        if (p / "test").is_dir() and (p / "train").is_dir():
            return p
        if (p / "MRSHaze" / "test").is_dir() and (p / "MRSHaze" / "train").is_dir():
            return p / "MRSHaze"

    # Search top-level subdirectories of /kaggle/input
    input_dir = Path("/kaggle/input")
    if input_dir.is_dir():
        for p in input_dir.rglob("test"):
            if p.is_dir() and (p.parent / "train").is_dir():
                return p.parent
    raise FileNotFoundError("Could not locate valid MRSHaze dataset under /kaggle/input")


def setup_repository(repo_url, branch, repo_dir):
    """Clone or checkout the specified branch in the working directory."""
    repo_dir = Path(repo_dir)
    if not (repo_dir / ".git").is_dir():
        print(f"Cloning {repo_url} (branch: {branch}) into {repo_dir}...", flush=True)
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend([repo_url, str(repo_dir)])
        subprocess.run(cmd, check=True)
    else:
        print(f"Repository exists at {repo_dir}. Fetching latest {branch}...", flush=True)
        subprocess.run(["git", "fetch", "--depth", "1", "origin", branch], cwd=repo_dir, check=True)
        subprocess.run(["git", "checkout", branch], cwd=repo_dir, check=True)
        subprocess.run(["git", "reset", "--hard", f"origin/{branch}"], cwd=repo_dir, check=True)

    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir).decode().strip()
    print(f"Repository ready at commit {git_hash} (branch: {branch})", flush=True)
    return repo_dir, git_hash


def run_negative_evaluation(repo_dir, weights_path, data_root, output_dir, device="cuda"):
    """Evaluate clear-pixel preservation: test if model alters haze-free GT images when guided by SAR."""
    import torch
    from PIL import Image
    import numpy as np

    sys.path.insert(0, str(repo_dir))
    from graytocolor.model import SARGrayColor
    from graytocolor.runtime import load_checkpoint, select_device
    from graytocolor.data import paired_records, load_rgb, load_sar, dataset_options

    checkpoint = load_checkpoint(weights_path)
    config = checkpoint["config"]
    dev = select_device(device)
    model = SARGrayColor(**config["model"]).to(dev).eval()
    model.load_state_dict(checkpoint["model"])

    records = paired_records(data_root, "test", require_gt=True)
    # Subsample 100 test images for negative evaluation to keep runtime fast
    sample_records = records[:min(100, len(records))]
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alterations = []
    with torch.inference_mode():
        for r in sample_records:
            gt_img = load_rgb(r["GT"])[None].to(dev)
            sar_img = load_sar(r["SAR"], *dataset_options(config).values())[None].to(dev)
            # Pass clean GT as input
            pred = model(gt_img, sar_img, structure_only=False)
            diff = (pred["rgb"] - gt_img).abs().mean().item()
            alterations.append(diff)

    mean_maa = float(np.mean(alterations))
    result = {
        "samples_evaluated": len(alterations),
        "mean_absolute_alteration_maa": mean_maa,
        "description": "Mean Absolute Alteration (L1 difference) when feeding clean GT + SAR through the dehazing network. Lower is better (0 = perfect preservation of clear pixels)."
    }
    (out_dir / "negative_eval.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"[Negative Eval] Mean Absolute Alteration (MAA) on clean test samples: {mean_maa:.5f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-url", default="https://github.com/sumire25/SARguidedRSID.git")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--repo-dir", default="/kaggle/working/SARguidedRSID")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    start_time = time.time()
    print("=== Starting Kaggle Experiment Worker ===", flush=True)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # 1. Discover dataset
    data_root = Path(args.data_root) if args.data_root else find_mrshaze_root()
    print(f"Discovered MRSHaze root: {data_root}", flush=True)

    # 2. Setup repository branch
    repo_dir, git_hash = setup_repository(args.repo_url, args.branch, args.repo_dir)

    # 3. Resolve config
    config_path = repo_dir / args.config
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    print(f"Using configuration: {config_path}", flush=True)

    # 4. Execute training
    run_dir = Path("/kaggle/working/runs") / Path(args.config).stem
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        print("\n--- Starting Training ---", flush=True)
        train_cmd = [
            sys.executable, "-u", "train.py",
            "--config", str(config_path)
        ]
        subprocess.run(train_cmd, cwd=repo_dir, check=True)
        print("--- Training Completed Successfully ---\n", flush=True)

    # 5. Execute Test Evaluation
    weights_path = run_dir / "best.pt"
    if not weights_path.is_file():
        weights_path = run_dir / "last.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(f"No checkpoint found at {run_dir}")

    print(f"\n--- Running Test Inference with weights {weights_path} ---", flush=True)
    results_dir = Path("/kaggle/working/results") / Path(args.config).stem
    test_out = results_dir / "test"
    test_out.mkdir(parents=True, exist_ok=True)

    infer_cmd = [
        sys.executable, "-u", "infer.py",
        "--weights", str(weights_path),
        "--data-root", str(data_root),
        "--split", "test",
        "--output-dir", str(test_out),
        "--device", args.device,
        "--save-aux"
    ]
    subprocess.run(infer_cmd, cwd=repo_dir, check=True)

    print("\n--- Running Standard 8-bit PNG Evaluation against GT ---", flush=True)
    eval_cmd = [
        sys.executable, "evaluate.py",
        "--predictions", str(test_out / "rgb"),
        "--targets", str(data_root / "test" / "GT"),
        "--output-dir", str(test_out / "png_metrics"),
        "--device", args.device
    ]
    subprocess.run(eval_cmd, cwd=repo_dir, check=True)

    # 6. Execute Negative Sample Evaluation (Clear-pixel alteration)
    print("\n--- Running Negative Sample (Clean Image) Alteration Evaluation ---", flush=True)
    neg_results = run_negative_evaluation(repo_dir, weights_path, data_root, results_dir / "negative", device=args.device)

    # 7. Package final summary & sample visual artifacts
    elapsed_hours = (time.time() - start_time) / 3600.0
    test_summary_file = test_out / "png_metrics" / "summary.json"
    test_metrics = json.loads(test_summary_file.read_text()) if test_summary_file.is_file() else {}

    final_summary = {
        "experiment": Path(args.config).stem,
        "branch": args.branch,
        "git_commit": git_hash,
        "config_file": str(args.config),
        "total_elapsed_hours": round(elapsed_hours, 3),
        "test_metrics": test_metrics.get("mean", {}),
        "negative_alteration": neg_results,
    }
    summary_path = results_dir / "final_experiment_summary.json"
    summary_path.write_text(json.dumps(final_summary, indent=2) + "\n")
    print("\n=== FINAL EXPERIMENT SUMMARY ===", flush=True)
    print(json.dumps(final_summary, indent=2), flush=True)

    # Copy 5 representative sample images (hazy, sar, rgb output, gt) to visuals/
    visuals_dir = results_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    pred_images = sorted((test_out / "rgb").glob("*.png"))[:5]
    for p in pred_images:
        name = p.name
        shutil.copy2(p, visuals_dir / f"pred_{name}")
        shutil.copy2(data_root / "test" / "hazy" / name, visuals_dir / f"hazy_{name}")
        shutil.copy2(data_root / "test" / "SAR" / name, visuals_dir / f"sar_{name}")
        shutil.copy2(data_root / "test" / "GT" / name, visuals_dir / f"gt_{name}")
    print(f"Sample visuals saved to {visuals_dir}", flush=True)


if __name__ == "__main__":
    main()
