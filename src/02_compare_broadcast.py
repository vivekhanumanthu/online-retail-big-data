from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from typing import Dict, List


def run_pipeline(mode: str, bronze_path: str, gold_root: str) -> Dict[str, str]:
    use_broadcast = mode == "broadcast"
    gold_path = f"{gold_root}/{mode}"
    plan_output = f"artifacts/spark_ui/02_pipeline_plan_{mode}.txt"

    cmd = [
        "python3",
        "src/02_pipeline.py",
        "--bronze_path",
        bronze_path,
        "--gold_path",
        gold_path,
        "--save_plan",
        "--plan_output",
        plan_output,
    ]
    if use_broadcast:
        cmd.append("--use_broadcast")

    start = time.perf_counter()
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    runtime_sec = time.perf_counter() - start

    return {
        "mode": mode,
        "success": "1" if proc.returncode == 0 else "0",
        "runtime_sec": f"{runtime_sec:.3f}",
        "gold_path": gold_path,
        "plan_output": plan_output,
        "stdout_last_line": (proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""),
        "stderr_last_line": (proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pipeline with and without broadcast joins")
    parser.add_argument("--bronze_path", type=str, default="outputs/bronze/online_retail")
    parser.add_argument("--gold_root", type=str, default="outputs/gold_compare")
    parser.add_argument("--summary_csv", type=str, default="outputs/tableau/broadcast_compare_csv/summary.csv")
    args = parser.parse_args()

    if not os.path.exists(args.bronze_path):
        raise FileNotFoundError(
            f"Bronze path not found: {args.bronze_path}. Run src/01_ingest.py first or pass correct --bronze_path."
        )

    os.makedirs(os.path.dirname(args.summary_csv), exist_ok=True)

    rows: List[Dict[str, str]] = []
    for mode in ["broadcast", "no_broadcast"]:
        rows.append(run_pipeline(mode, args.bronze_path, args.gold_root))

    with open(args.summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "success",
                "runtime_sec",
                "gold_path",
                "plan_output",
                "stdout_last_line",
                "stderr_last_line",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Comparison complete. Summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
