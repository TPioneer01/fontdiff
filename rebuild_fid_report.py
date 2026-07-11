"""Rebuild detailed FID report from an existing batch_webui run.

Usage:
    python rebuild_fid_report.py --run_root outputs/batch_webui/20260601_120000
    python rebuild_fid_report.py --tmp_root outputs/batch_webui/20260601_120000/category_fid_tmp
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import List, Dict

from evaluation import BatchEvaluator


def _write_fid_report_csv(summary_path: Path, fid_reports: List[Dict[str, object]]) -> None:
    with summary_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["level", "category", "style", "model", "fid", "source_dir", "target_dir", "status", "error"])
        for report in fid_reports:
            writer.writerow([
                report.get("level", ""),
                report.get("category", ""),
                report.get("style", ""),
                report.get("model", ""),
                report.get("fid", ""),
                report.get("source_dir", ""),
                report.get("target_dir", ""),
                report.get("status", ""),
                report.get("error", ""),
            ])


def _resolve_tmp_root(run_root: Path, tmp_root: Path | None) -> Path:
    if tmp_root is not None:
        return tmp_root
    return run_root / "category_fid_tmp"


def _build_fid_reports(tmp_root: Path, evaluator: BatchEvaluator) -> List[Dict[str, object]]:
    fid_reports: List[Dict[str, object]] = []

    if not tmp_root.exists():
        raise FileNotFoundError(f"FID tmp directory not found: {tmp_root}")

    for category_dir in sorted([path for path in tmp_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower()):
        target_dir = category_dir / "target"
        modela_dir = category_dir / "modela"
        modelb_dir = category_dir / "modelb"

        if not target_dir.exists():
            fid_reports.append({
                "level": "category",
                "category": category_dir.name,
                "style": "",
                "model": "modela",
                "fid": "",
                "source_dir": str(modela_dir),
                "target_dir": str(target_dir),
                "status": "error",
                "error": "missing target dir",
            })
            fid_reports.append({
                "level": "category",
                "category": category_dir.name,
                "style": "",
                "model": "modelb",
                "fid": "",
                "source_dir": str(modelb_dir),
                "target_dir": str(target_dir),
                "status": "error",
                "error": "missing target dir",
            })
            continue

        for model_name, source_dir in (("modela", modela_dir), ("modelb", modelb_dir)):
            if not source_dir.exists():
                fid_reports.append({
                    "level": "category",
                    "category": category_dir.name,
                    "style": "",
                    "model": model_name,
                    "fid": "",
                    "source_dir": str(source_dir),
                    "target_dir": str(target_dir),
                    "status": "error",
                    "error": "missing source dir",
                })
                continue

            try:
                fid_value = float(evaluator.calculate_fid(source_dir, target_dir))
                fid_reports.append({
                    "level": "category",
                    "category": category_dir.name,
                    "style": "",
                    "model": model_name,
                    "fid": fid_value,
                    "source_dir": str(source_dir),
                    "target_dir": str(target_dir),
                    "status": "ok",
                    "error": "",
                })
            except Exception as error:
                fid_reports.append({
                    "level": "category",
                    "category": category_dir.name,
                    "style": "",
                    "model": model_name,
                    "fid": "",
                    "source_dir": str(source_dir),
                    "target_dir": str(target_dir),
                    "status": "error",
                    "error": str(error),
                })

    return fid_reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild FID report from existing category_fid_tmp data.")
    parser.add_argument("--run_root", type=str, default=None, help="Existing run root containing category_fid_tmp")
    parser.add_argument("--tmp_root", type=str, default=None, help="Direct path to category_fid_tmp")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: <run_root>/fid_report.csv)")
    parser.add_argument("--cleanup", action="store_true", help="Remove category_fid_tmp after generating the CSV")
    parser.add_argument("--device", type=str, default=None, help="Evaluation device, default auto-select")
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve() if args.run_root else None
    tmp_root = Path(args.tmp_root).expanduser().resolve() if args.tmp_root else None

    if tmp_root is None:
        if run_root is None:
            parser.error("Provide either --run_root or --tmp_root")
        tmp_root = _resolve_tmp_root(run_root, None)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        if run_root is None:
            output_path = tmp_root.parent / "fid_report.csv"
        else:
            output_path = run_root / "fid_report.csv"

    evaluator = BatchEvaluator(device=args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    fid_reports = _build_fid_reports(tmp_root, evaluator)
    _write_fid_report_csv(output_path, fid_reports)

    print(f"FID report written to: {output_path}")
    print(f"Processed categories: {len({row['category'] for row in fid_reports})}")

    if args.cleanup and tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"Removed temporary directory: {tmp_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
