"""Comparison script to aggregate metrics across ResNet, VGG, and MobileNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def load_model_run(run_dir: Path) -> Dict[str, Any] | None:
    metrics_file = run_dir / "metrics.json"
    if not metrics_file.is_file():
        return None
    try:
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
        return data
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")
        return None


def compare_models(runs_root: Path) -> None:
    runs = [d for d in runs_root.iterdir() if d.is_dir()]
    results: List[Dict[str, Any]] = []

    for r in runs:
        info = load_model_run(r)
        if info is not None:
            results.append(info)

    if not results:
        print(f"No completed model runs found under {runs_root}")
        return

    print("=" * 105)
    print(f"{'Architecture':<12} | {'Variant':<18} | {'Params (M)':<10} | {'Latency (ms)':<12} | {'Val MAE':<10} | {'Test MAE':<10} | {'Test RMSE':<10} | {'Test R²':<8} | {'Pearson r':<8}")
    print("=" * 105)

    best_run = None
    lowest_mae = float("inf")

    for res in sorted(results, key=lambda x: x.get("test_metrics", {}).get("mae_g_dl", 999.0)):
        arch = res.get("architecture", "unknown")
        variant = res.get("variant", "unknown")
        params = res.get("parameters", {}).get("trainable", 0) / 1e6
        lat = res.get("inference_latency_ms_per_image", 0.0)
        v_mae = res.get("best_val_mae_g_dl", 0.0)
        t_metrics = res.get("test_metrics", {})
        t_mae = t_metrics.get("mae_g_dl", 0.0)
        t_rmse = t_metrics.get("rmse_g_dl", 0.0)
        t_r2 = t_metrics.get("r2", 0.0)
        t_pr = t_metrics.get("pearson_r", 0.0)

        print(f"{arch:<12} | {variant:<18} | {params:<10.2f} | {lat:<12.1f} | {v_mae:<10.3f} | {t_mae:<10.3f} | {t_rmse:<10.3f} | {t_r2:<8.3f} | {t_pr:<8.3f}")

        if t_mae < lowest_mae:
            lowest_mae = t_mae
            best_run = res

    print("=" * 105)
    if best_run:
        print(f"\n[WINNER]: {best_run['architecture']} ({best_run['variant']}) with Test MAE of {lowest_mae:.3f} g/dL.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained Hemoglobin predictor architectures.")
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    args = parser.parse_args()

    compare_models(args.runs_dir)


if __name__ == "__main__":
    main()

