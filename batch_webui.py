"""独立的批量推理与评价 WebUI。

该入口尽量不改动现有 `gradio_app.py`，只复用现有模型加载、采样和评价能力，
面向大批量目录级任务：一次加载模型、按批送入 GPU、按批保存结果、任务结束后统一评价。
"""

from __future__ import annotations

import copy
import csv
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as torch_F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import shutil
from datetime import timedelta

from evaluation import BatchEvaluator
from sample import arg_parse, load_fontdiffuer_pipeline, resolve_runtime_device
from utils import is_char_in_font, load_ttf, ttf2im


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
EVAL_METRICS = ["L1", "L2", "RMSE", "PSNR", "SSIM", "LPIPS"]
LOG_INTERVAL_SECONDS = 30


def _configure_torch_runtime() -> None:
    if not os.environ.get("XDG_RUNTIME_DIR"):
        runtime_dir = Path(tempfile.gettempdir()) / "fontdiff_xdg_runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_RUNTIME_DIR"] = str(runtime_dir)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _load_css_or_empty() -> str:
    css_path = Path(__file__).with_name("gradio_app.css")
    if css_path.exists():
        return css_path.read_text(encoding="utf-8")
    return ""


def _normalize_metrics(selected_metrics: Sequence[str] | None) -> List[str]:
    if not selected_metrics:
        return list(EVAL_METRICS)
    normalized: List[str] = []
    for metric in selected_metrics:
        metric_name = str(metric).strip().upper()
        if metric_name in EVAL_METRICS and metric_name not in normalized:
            normalized.append(metric_name)
    return normalized or list(EVAL_METRICS)


def _safe_name(text: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in str(text).strip())
    cleaned = cleaned.strip("_")
    return cleaned or "sample"


@lru_cache(maxsize=8)
def _cached_font(ttf_path: str, font_size: int):
    return load_ttf(ttf_path=ttf_path, fsize=font_size)


def _read_source_characters(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        return []

    characters: List[str] = []
    for line in txt_path.read_text(encoding="utf-8-sig").splitlines():
        for chunk in line.replace("，", ",").split(","):
            character = chunk.strip()
            if character:
                characters.append(character)
    return characters


def _discover_style_directories(category_dir: Path) -> List[Path]:
    if not category_dir.exists():
        return []
    return sorted([path for path in category_dir.iterdir() if path.is_dir()], key=lambda path: path.name.lower())


def _extract_target_character(image_path: Path) -> str:
    stem = image_path.stem.strip()
    if "+" in stem:
        _, content_name = stem.split("+", 1)
        return content_name.strip()
    return stem


def _build_target_lookup(target_images: Sequence[Path]) -> Tuple[dict, Path | None]:
    lookup = {}
    style_reference_path = target_images[0] if target_images else None

    for target_path in target_images:
        character = _extract_target_character(target_path)
        priority = 0 if target_path.stem == character else 1
        existing = lookup.get(character)
        if existing is None or priority < existing[0]:
            lookup[character] = (priority, target_path)

    resolved_lookup = {character: path for character, (_, path) in lookup.items()}
    return resolved_lookup, style_reference_path


def _discover_eval_samples(val_root: Path, source_chars_root: Path) -> List[dict]:
    if not val_root.exists() or not val_root.is_dir():
        return []

    category_dirs = [path for path in val_root.iterdir() if path.is_dir()]
    category_dirs.sort(key=lambda path: path.name.lower())

    samples: List[dict] = []
    for category_dir in category_dirs:
        source_txt_path = source_chars_root / f"{category_dir.name}.txt"
        source_characters = _read_source_characters(source_txt_path)
        if not source_characters:
            continue

        for style_dir in _discover_style_directories(category_dir):
            target_images = _list_image_files(style_dir)
            if not target_images:
                continue

            target_by_content, style_reference_path = _build_target_lookup(target_images)

            for index, character in enumerate(source_characters):
                target_path = target_by_content.get(character)

                if target_path is None:
                    continue

                samples.append(
                    {
                        "category": category_dir.name,
                        "style": style_dir.name,
                        "character": character,
                        "source_txt": source_txt_path,
                        "style_reference_path": style_reference_path,
                        "target_path": target_path,
                        "sample_key": f"{category_dir.name}/{style_dir.name}/{index:04d}_{character}",
                    }
                )

    return samples


def _render_character_image(ttf_path: str, character: str, font_size: int = 128):
    if not is_char_in_font(font_path=ttf_path, char=character):
        return None

    font = _cached_font(ttf_path, font_size)
    image = ttf2im(font=font, char=character, fsize=font_size)
    return image.convert("RGB") if image is not None else None


def _ensure_rgb_pil_image(image, resolution=None):
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    image = image.convert("RGB")
    if resolution is not None and image.size != (resolution, resolution):
        image = image.resize((resolution, resolution), Image.BILINEAR)
    return image


def _load_label_font(font_size: int = 20):
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        return ImageFont.load_default()


def _build_labelled_strip(images: Sequence[Image.Image], labels: Sequence[str], label_height: int = 32) -> Image.Image:
    images = [_ensure_rgb_pil_image(image) for image in images]
    resolution = images[0].size[0]
    canvas = Image.new("RGB", (resolution * len(images), resolution + label_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _load_label_font(max(16, label_height - 8))

    for index, (image, label) in enumerate(zip(images, labels)):
        if image.size != (resolution, resolution):
            image = image.resize((resolution, resolution), Image.BILINEAR)
        offset_x = resolution * index
        canvas.paste(image, (offset_x, 0))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_x = offset_x + max(0, (resolution - text_width) // 2)
        text_y = resolution + max(0, (label_height - text_height) // 2) - 1
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
    return canvas


def _build_three_way_panel(target_image: Image.Image, modela_image: Image.Image, modelb_image: Image.Image):
    target_image = _ensure_rgb_pil_image(target_image)
    modela_image = _ensure_rgb_pil_image(modela_image)
    modelb_image = _ensure_rgb_pil_image(modelb_image)

    return _build_labelled_strip(
        [target_image, modela_image, modelb_image],
        ["target", "a", "b"],
    )


def _format_time(seconds: float) -> str:
    if seconds is None:
        return "unknown"
    try:
        return str(timedelta(seconds=int(seconds)))
    except Exception:
        return f"{seconds:.1f}s"


def _format_bytes(num_bytes: float) -> str:
    if num_bytes is None:
        return "unknown"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0


def _build_output_style_roots(output_root: Path, category: str, style: str) -> dict:
    style_root = output_root / category / style
    return {
        "target": style_root / "target",
        "modela": style_root / "modela",
        "modelb": style_root / "modelb",
        "compare": style_root / "compare",
    }


def _list_image_files(directory: str | os.PathLike[str]) -> List[Path]:
    directory_path = Path(directory)
    if not directory_path.exists():
        return []
    files = [path for path in directory_path.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    return sorted(files, key=lambda path: path.name.lower())


def _pair_files_by_mode(
    content_files: Sequence[Path],
    reference_files: Sequence[Path],
    pairing_mode: str,
) -> Tuple[List[Tuple[Path, Path]], str | None]:
    pairing_mode = str(pairing_mode).strip().lower()
    if pairing_mode == "same_name":
        reference_map = {path.stem: path for path in reference_files}
        pairs = []
        missing = []
        for content_path in content_files:
            reference_path = reference_map.get(content_path.stem)
            if reference_path is None:
                missing.append(content_path.name)
                continue
            pairs.append((content_path, reference_path))
        note = None
        if missing:
            note = f"有 {len(missing)} 个 content 文件未找到同名 reference，已跳过。"
        return pairs, note

    pair_count = min(len(content_files), len(reference_files))
    note = None
    if len(content_files) != len(reference_files):
        note = f"文件数量不一致，当前仅按排序后前 {pair_count} 对进行处理。"
    return list(zip(content_files[:pair_count], reference_files[:pair_count])), note


def _load_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image_file:
        return image_file.convert("RGB")


def _save_compare_panel(output_dir: Path, stem: str, target_image: Image.Image, modela_image: Image.Image, modelb_image: Image.Image) -> Path:
    panel = _build_three_way_panel(target_image, modela_image, modelb_image)
    panel_path = output_dir / f"compare_{stem}.png"
    panel.save(panel_path)
    return panel_path


def _save_generated_image(output_dir: Path, stem: str, image: Image.Image) -> Path:
    image_path = output_dir / f"gen_{stem}.png"
    image.save(image_path)
    return image_path


def _save_target_image(output_dir: Path, stem: str, image: Image.Image) -> Path:
    image_path = output_dir / f"target_{stem}.png"
    image.save(image_path)
    return image_path


def _remove_small_dark_components(image: Image.Image, min_area_ratio: float = 0.0015, dark_threshold: int = 240) -> Image.Image:
    """Remove tiny dark speckles from very small generated images.

    This is tuned for glyph-like outputs on 64x64 crops. It only removes small
    isolated connected components in the dark-pixel mask and keeps larger stroke
    structures intact as much as possible.
    """
    image = _ensure_rgb_pil_image(image)
    width, height = image.size
    if max(width, height) > 64:
        return image

    rgb = np.array(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark_mask = (gray < dark_threshold).astype(np.uint8)
    if dark_mask.max() == 0:
        return image

    min_area = max(4, int(width * height * float(min_area_ratio)))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_mask, connectivity=8)

    cleaned_mask = np.zeros_like(dark_mask)
    for label_index in range(1, num_labels):
        area = stats[label_index, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == label_index] = 1

    cleaned_rgb = rgb.copy()
    cleaned_rgb[cleaned_mask == 0] = 255
    return Image.fromarray(cleaned_rgb)


def _extract_saved_stem(file_path: Path, prefix: str) -> str | None:
    name = file_path.stem
    if not name.startswith(prefix):
        return None
    return name[len(prefix):]


def _scan_completed_outputs(run_root: Path) -> dict[str, dict[str, Path]]:
    completed: dict[str, dict[str, Path]] = {}
    if not run_root.exists():
        return completed

    for target_path in run_root.glob("**/target/target_*.png"):
        stem = _extract_saved_stem(target_path, "target_")
        if not stem:
            continue

        modela_path = target_path.parent.parent / "modela" / f"gen_{stem}.png"
        modelb_path = target_path.parent.parent / "modelb" / f"gen_{stem}.png"
        compare_path = target_path.parent.parent / "compare" / f"compare_{stem}.png"
        if not (modela_path.exists() and modelb_path.exists() and compare_path.exists()):
            continue

        completed[stem] = {
            "target": target_path,
            "modela": modela_path,
            "modelb": modelb_path,
            "compare": compare_path,
        }

    return completed


def _write_summary_csv(summary_path: Path, results: dict, metadata: dict) -> None:
    with summary_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["section", "key", "value"])
        for key, value in metadata.items():
            writer.writerow(["meta", key, value])
        for metric_name, metric_stats in results.items():
            if not isinstance(metric_stats, dict):
                continue
            writer.writerow(["metric", metric_name, json.dumps(metric_stats, ensure_ascii=False)])



def _write_style_report_csv(summary_path: Path, style_reports: Sequence[dict]) -> None:
    with summary_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["category", "style", "model", "metric", "mean", "std", "min", "max"])
        for report in style_reports:
            for model_name in ("modela", "modelb"):
                model_metrics = report.get(model_name, {})
                if not isinstance(model_metrics, dict):
                    continue
                for metric_name, metric_stats in model_metrics.items():
                    if not isinstance(metric_stats, dict):
                        continue
                    writer.writerow([
                        report.get("category", ""),
                        report.get("style", ""),
                        model_name,
                        metric_name,
                        metric_stats.get("mean", ""),
                        metric_stats.get("std", ""),
                        metric_stats.get("min", ""),
                        metric_stats.get("max", ""),
                    ])


def _write_category_report_csv(summary_path: Path, category_reports: Sequence[dict]) -> None:
    with summary_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["category", "model", "metric", "mean", "std", "min", "max"]) 
        for report in category_reports:
            for model_name in ("modela", "modelb"):
                model_metrics = report.get(model_name, {})
                if not isinstance(model_metrics, dict):
                    continue
                for metric_name, metric_stats in model_metrics.items():
                    if not isinstance(metric_stats, dict):
                        continue
                    writer.writerow([
                        report.get("category", ""),
                        model_name,
                        metric_name,
                        metric_stats.get("mean", ""),
                        metric_stats.get("std", ""),
                        metric_stats.get("min", ""),
                        metric_stats.get("max", ""),
                    ])


def _write_fid_report_csv(summary_path: Path, fid_reports: Sequence[dict]) -> None:
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

def _collect_evaluation_pairs(samples: Sequence[dict], modela_paths: Sequence[Path], modelb_paths: Sequence[Path]) -> List[dict]:
    pair_count = min(len(samples), len(modela_paths), len(modelb_paths))
    evaluation_pairs = []
    for index in range(pair_count):
        sample = samples[index]
        evaluation_pairs.append(
            {
                "category": sample["category"],
                "style": sample["style"],
                "character": sample["character"],
                "target_path": sample["target_path"],
                "modela_path": modela_paths[index],
                "modelb_path": modelb_paths[index],
            }
        )
    return evaluation_pairs


def _build_evaluation_arrays(evaluation_pairs: Sequence[dict]) -> Tuple[List[Image.Image], List[Image.Image], List[Image.Image]]:
    target_arrays = []
    modela_arrays = []
    modelb_arrays = []

    for pair in evaluation_pairs:
        target_arrays.append(_load_image(Path(pair["target_path"])))
        modela_arrays.append(_load_image(Path(pair["modela_path"])))
        modelb_arrays.append(_load_image(Path(pair["modelb_path"])))

    return target_arrays, modela_arrays, modelb_arrays


def _evaluate_model_outputs(evaluator: BatchEvaluator, generated_arrays: Sequence[Image.Image], target_arrays: Sequence[Image.Image], metrics: Sequence[str]) -> dict:
    valid_count = min(len(generated_arrays), len(target_arrays))
    if valid_count == 0:
        return {}
    return evaluator.evaluate_batch(
        list(generated_arrays[:valid_count]),
        list(target_arrays[:valid_count]),
        metrics=list(metrics),
    )


def _evaluate_style_group(
    evaluator: BatchEvaluator,
    style_group: dict,
    metrics: Sequence[str],
) -> dict:
    non_fid_metrics = [metric for metric in metrics if str(metric).upper() != "FID"]
    target_paths = style_group.get("target_paths", [])
    modela_paths = style_group.get("modela_paths", [])
    modelb_paths = style_group.get("modelb_paths", [])

    target_arrays = [_load_image(Path(path)) for path in target_paths]
    modela_arrays = [_load_image(Path(path)) for path in modela_paths]
    modelb_arrays = [_load_image(Path(path)) for path in modelb_paths]

    style_dir_paths = style_group.get("style_dirs", {})
    target_dir = style_dir_paths.get("target")
    modela_dir = style_dir_paths.get("modela")
    modelb_dir = style_dir_paths.get("modelb")

    style_result = {
        "category": style_group.get("category"),
        "style": style_group.get("style"),
        "sample_count": min(len(target_arrays), len(modela_arrays), len(modelb_arrays)),
        "modela": _evaluate_model_outputs(evaluator, modela_arrays, target_arrays, non_fid_metrics),
        "modelb": _evaluate_model_outputs(evaluator, modelb_arrays, target_arrays, non_fid_metrics),
    }

    if target_dir and modela_dir and "FID" in [metric.upper() for metric in metrics]:
        try:
            fid_value = float(evaluator.calculate_fid(modela_dir, target_dir))
            style_result["modela"]["FID"] = {
                "mean": fid_value,
                "std": 0.0,
                "min": fid_value,
                "max": fid_value,
            }
            style_result.setdefault("fid_reports", []).append({
                "level": "style",
                "category": style_group.get("category"),
                "style": style_group.get("style"),
                "model": "modela",
                "fid": fid_value,
                "source_dir": str(modela_dir),
                "target_dir": str(target_dir),
                "status": "ok",
                "error": "",
            })
        except Exception as error:
            style_result.setdefault("errors", {})["modela_fid"] = str(error)
            style_result.setdefault("fid_reports", []).append({
                "level": "style",
                "category": style_group.get("category"),
                "style": style_group.get("style"),
                "model": "modela",
                "fid": "",
                "source_dir": str(modela_dir) if modela_dir else "",
                "target_dir": str(target_dir) if target_dir else "",
                "status": "error",
                "error": str(error),
            })

    if target_dir and modelb_dir and "FID" in [metric.upper() for metric in metrics]:
        try:
            fid_value = float(evaluator.calculate_fid(modelb_dir, target_dir))
            style_result["modelb"]["FID"] = {
                "mean": fid_value,
                "std": 0.0,
                "min": fid_value,
                "max": fid_value,
            }
            style_result.setdefault("fid_reports", []).append({
                "level": "style",
                "category": style_group.get("category"),
                "style": style_group.get("style"),
                "model": "modelb",
                "fid": fid_value,
                "source_dir": str(modelb_dir),
                "target_dir": str(target_dir),
                "status": "ok",
                "error": "",
            })
        except Exception as error:
            style_result.setdefault("errors", {})["modelb_fid"] = str(error)
            style_result.setdefault("fid_reports", []).append({
                "level": "style",
                "category": style_group.get("category"),
                "style": style_group.get("style"),
                "model": "modelb",
                "fid": "",
                "source_dir": str(modelb_dir) if modelb_dir else "",
                "target_dir": str(target_dir) if target_dir else "",
                "status": "error",
                "error": str(error),
            })

    return style_result


def _prepare_batch_tensors(
    content_images: Sequence[Image.Image],
    style_images: Sequence[Image.Image],
    content_size: Tuple[int, int],
    style_size: Tuple[int, int],
    edge_low: int,
    edge_high: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Image.Image], List[Image.Image]]:
    content_transform = transforms.Compose(
        [
            transforms.Resize(content_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    style_transform = transforms.Compose(
        [
            transforms.Resize(style_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    content_tensors = []
    style_tensors = []
    content_edges = []
    style_edges = []
    content_pils: List[Image.Image] = []
    style_pils: List[Image.Image] = []

    for content_image, reference_image in zip(content_images, style_images):
        content_pils.append(content_image)
        style_pils.append(reference_image)

        content_tensors.append(content_transform(content_image))
        style_tensors.append(style_transform(reference_image))

        content_edge_np = cv2.Canny(np.array(content_image.convert("L")), threshold1=edge_low, threshold2=edge_high)
        style_edge_np = cv2.Canny(np.array(reference_image.convert("L")), threshold1=edge_low, threshold2=edge_high)
        content_edge = torch.from_numpy(content_edge_np).float().unsqueeze(0).unsqueeze(0) / 255.0
        style_edge = torch.from_numpy(style_edge_np).float().unsqueeze(0).unsqueeze(0) / 255.0
        content_edge = torch_F.interpolate(content_edge, size=content_size, mode="nearest")
        style_edge = torch_F.interpolate(style_edge, size=style_size, mode="nearest")
        content_edges.append(content_edge)
        style_edges.append(style_edge)

    content_tensor = torch.stack(content_tensors, dim=0)
    style_tensor = torch.stack(style_tensors, dim=0)
    content_edge_tensor = torch.cat(content_edges, dim=0)
    style_edge_tensor = torch.cat(style_edges, dim=0)

    return content_tensor, style_tensor, content_edge_tensor, style_edge_tensor, content_pils, style_pils


@dataclass
class BatchWebUIConfig:
    val_root: str
    source_chars_root: str
    ttf_path: str
    ckpt_dir_a: str
    ckpt_dir_b: str
    device_mode: str
    resume_run_root: str
    output_root: str
    batch_size: int
    num_inference_steps: int
    guidance_scale: float
    compute_metrics: bool
    save_panels: bool
    enable_small_image_denoise: bool
    metrics: List[str]


class BatchInferenceRunner:
    def __init__(self, base_args):
        self.base_args = base_args
        self._pipeline_cache: dict[tuple[str, str], object] = {}

    def _build_run_args(self, resolved_device: str, ckpt_dir: str, num_inference_steps: int, guidance_scale: float):
        run_args = copy.deepcopy(self.base_args)
        run_args.device = resolved_device
        run_args.ckpt_dir = ckpt_dir
        run_args.num_inference_steps = int(num_inference_steps)
        run_args.guidance_scale = float(guidance_scale)
        run_args.save_image = False
        run_args.demo = True
        return run_args

    def _load_pipeline(self, resolved_device: str, ckpt_dir: str):
        cache_key = (resolved_device, ckpt_dir)
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]

        run_args = self._build_run_args(
            resolved_device=resolved_device,
            ckpt_dir=ckpt_dir,
            num_inference_steps=self.base_args.num_inference_steps,
            guidance_scale=self.base_args.guidance_scale,
        )
        pipe = load_fontdiffuer_pipeline(args=run_args)
        self._pipeline_cache[cache_key] = pipe
        return pipe

    def run(self, config: BatchWebUIConfig, progress=gr.Progress()):
        val_root = Path(config.val_root)
        source_chars_root = Path(config.source_chars_root)
        samples = _discover_eval_samples(val_root, source_chars_root)

        if not samples:
            return (
                "### 没有发现可处理的样本。请检查 val 目录和 txt 目录。",
                "### 性能统计\n- 尚未运行",
                json.dumps({"error": "no samples"}, ensure_ascii=False, indent=2),
                [],
                "",
                None,
                None,
            )

        resolved_device, device_message = resolve_runtime_device(config.device_mode)
        pipe_a = self._load_pipeline(resolved_device=resolved_device, ckpt_dir=config.ckpt_dir_a)
        pipe_b = self._load_pipeline(resolved_device=resolved_device, ckpt_dir=config.ckpt_dir_b)

        if config.resume_run_root:
            run_root = Path(config.resume_run_root)
        else:
            run_root = Path(config.output_root or Path("outputs") / "batch_webui")
            run_root = run_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root.mkdir(parents=True, exist_ok=True)
        output_roots = {
            "target": run_root / "target",
            "modela": run_root / "modela",
            "modelb": run_root / "modelb",
            "compare": run_root / "compare",
        }
        for root_path in output_roots.values():
            root_path.mkdir(parents=True, exist_ok=True)

        run_args = self._build_run_args(
            resolved_device=resolved_device,
            ckpt_dir=config.ckpt_dir_a,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
        )

        total_pairs = len(samples)
        modela_paths: List[Path] = []
        modelb_paths: List[Path] = []
        target_paths: List[Path] = []
        preview_gallery = []
        batch_count = max(1, int(config.batch_size))
        skipped_samples: List[str] = []
        processed_samples: List[dict] = []
        category_counter = {}
        note = None
        font_size = int(getattr(run_args, "content_image_size", (128, 128))[0])
        style_groups: dict[tuple[str, str], dict] = {}
        batch_durations: List[float] = []
        completed_outputs = _scan_completed_outputs(run_root)
        completed_sample_keys = set()

        for sample in samples:
            stem = _safe_name(f"{sample['category']}__{sample['style']}__{sample['character']}")
            saved_outputs = completed_outputs.get(stem)
            if saved_outputs is None:
                continue
            completed_sample_keys.add(sample["sample_key"])
            processed_samples.append(sample)
            category_counter[sample["category"]] = category_counter.get(sample["category"], 0) + 1
            target_paths.append(saved_outputs["target"])
            modela_paths.append(saved_outputs["modela"])
            modelb_paths.append(saved_outputs["modelb"])

            style_key = (sample["category"], sample["style"])
            if style_key not in style_groups:
                style_groups[style_key] = {
                    "category": sample["category"],
                    "style": sample["style"],
                    "style_dirs": _build_output_style_roots(run_root, sample["category"], sample["style"]),
                    "target_paths": [],
                    "modela_paths": [],
                    "modelb_paths": [],
                }
            style_groups[style_key]["target_paths"].append(saved_outputs["target"])
            style_groups[style_key]["modela_paths"].append(saved_outputs["modela"])
            style_groups[style_key]["modelb_paths"].append(saved_outputs["modelb"])

        resumed_sample_count = len(completed_sample_keys)

        if resolved_device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        start_time = time.perf_counter()
        active_start_time = time.perf_counter()
        last_log_time = start_time
        processed_count = resumed_sample_count
        active_processed_count = 0
        for batch_index, start in enumerate(range(0, total_pairs, batch_count), start=1):
            batch_samples = [sample for sample in samples[start:start + batch_count] if sample["sample_key"] not in completed_sample_keys]
            batch_start_time = time.perf_counter()

            # 计算当前 ETA 并在进度描述中显示
            elapsed = time.perf_counter() - active_start_time
            processed = active_processed_count
            rate = (processed / elapsed) if elapsed > 0 else 0.0
            remaining = max(0, total_pairs - resumed_sample_count - processed)
            eta = (remaining / rate) if rate > 0 else None
            eta_str = _format_time(eta)

            progress(
                (resumed_sample_count + processed) / total_pairs,
                desc=f"准备第 {batch_index} 批 / 共 {((total_pairs + batch_count - 1) // batch_count)} 批，批内 {len(batch_samples)}，已续跑 {resumed_sample_count}，ETA: {eta_str}",
            )

            batch_content_images = []
            batch_style_images = []
            batch_target_images = []
            batch_valid_samples = []

            for sample in batch_samples:
                content_image = _render_character_image(config.ttf_path, sample["character"], font_size=font_size)
                if content_image is None:
                    skipped_samples.append(f'{sample["sample_key"]} -> 字符不在字体中或无法渲染')
                    continue

                style_image = _load_image(sample["style_reference_path"])
                target_image = _load_image(sample["target_path"])

                batch_content_images.append(content_image)
                batch_style_images.append(style_image)
                batch_target_images.append(target_image)
                batch_valid_samples.append(sample)
                processed_samples.append(sample)

                category_counter[sample["category"]] = category_counter.get(sample["category"], 0) + 1

            if not batch_valid_samples:
                continue

            content_tensor, style_tensor, content_edges, style_edges, content_pils, style_pils = _prepare_batch_tensors(
                content_images=batch_content_images,
                style_images=batch_style_images,
                content_size=tuple(run_args.content_image_size),
                style_size=tuple(run_args.style_image_size),
                edge_low=int(run_args.edge_canny_low),
                edge_high=int(run_args.edge_canny_high),
            )

            content_tensor = content_tensor.to(resolved_device, non_blocking=True)
            style_tensor = style_tensor.to(resolved_device, non_blocking=True)
            content_edges = content_edges.to(resolved_device, non_blocking=True)
            style_edges = style_edges.to(resolved_device, non_blocking=True)

            with torch.inference_mode():
                if resolved_device.startswith("cuda"):
                    autocast_context = torch.autocast(device_type="cuda", dtype=torch.float16)
                else:
                    autocast_context = None

                if autocast_context is None:
                    generated_images_a = pipe_a.generate(
                        content_images=content_tensor,
                        style_images=style_tensor,
                        content_edges=content_edges,
                        style_edges=style_edges,
                        batch_size=len(batch_valid_samples),
                        order=run_args.order,
                        num_inference_step=run_args.num_inference_steps,
                        content_encoder_downsample_size=run_args.content_encoder_downsample_size,
                        t_start=run_args.t_start,
                        t_end=run_args.t_end,
                        dm_size=run_args.content_image_size,
                        algorithm_type=run_args.algorithm_type,
                        skip_type=run_args.skip_type,
                        method=run_args.method,
                        correcting_x0_fn=run_args.correcting_x0_fn,
                    )
                    generated_images_b = pipe_b.generate(
                        content_images=content_tensor,
                        style_images=style_tensor,
                        content_edges=content_edges,
                        style_edges=style_edges,
                        batch_size=len(batch_valid_samples),
                        order=run_args.order,
                        num_inference_step=run_args.num_inference_steps,
                        content_encoder_downsample_size=run_args.content_encoder_downsample_size,
                        t_start=run_args.t_start,
                        t_end=run_args.t_end,
                        dm_size=run_args.content_image_size,
                        algorithm_type=run_args.algorithm_type,
                        skip_type=run_args.skip_type,
                        method=run_args.method,
                        correcting_x0_fn=run_args.correcting_x0_fn,
                    )
                else:
                    with autocast_context:
                        generated_images_a = pipe_a.generate(
                            content_images=content_tensor,
                            style_images=style_tensor,
                            content_edges=content_edges,
                            style_edges=style_edges,
                            batch_size=len(batch_valid_samples),
                            order=run_args.order,
                            num_inference_step=run_args.num_inference_steps,
                            content_encoder_downsample_size=run_args.content_encoder_downsample_size,
                            t_start=run_args.t_start,
                            t_end=run_args.t_end,
                            dm_size=run_args.content_image_size,
                            algorithm_type=run_args.algorithm_type,
                            skip_type=run_args.skip_type,
                            method=run_args.method,
                            correcting_x0_fn=run_args.correcting_x0_fn,
                        )
                        generated_images_b = pipe_b.generate(
                            content_images=content_tensor,
                            style_images=style_tensor,
                            content_edges=content_edges,
                            style_edges=style_edges,
                            batch_size=len(batch_valid_samples),
                            order=run_args.order,
                            num_inference_step=run_args.num_inference_steps,
                            content_encoder_downsample_size=run_args.content_encoder_downsample_size,
                            t_start=run_args.t_start,
                            t_end=run_args.t_end,
                            dm_size=run_args.content_image_size,
                            algorithm_type=run_args.algorithm_type,
                            skip_type=run_args.skip_type,
                            method=run_args.method,
                            correcting_x0_fn=run_args.correcting_x0_fn,
                        )

            batch_durations.append(time.perf_counter() - batch_start_time)

            for sample_index, (sample, modela_image, modelb_image, content_pil, style_pil, target_pil) in enumerate(
                zip(batch_valid_samples, generated_images_a, generated_images_b, content_pils, style_pils, batch_target_images),
                start=start + 1,
            ):
                stem = _safe_name(f"{sample['category']}__{sample['style']}__{sample['character']}")
                style_output_roots = _build_output_style_roots(run_root, sample["category"], sample["style"])
                for root_path in style_output_roots.values():
                    root_path.mkdir(parents=True, exist_ok=True)

                style_key = (sample["category"], sample["style"])
                if style_key not in style_groups:
                    style_groups[style_key] = {
                        "category": sample["category"],
                        "style": sample["style"],
                        "style_dirs": style_output_roots,
                        "target_paths": [],
                        "modela_paths": [],
                        "modelb_paths": [],
                    }

                target_path = _save_target_image(style_output_roots["target"], stem, target_pil)
                modela_path = _save_generated_image(style_output_roots["modela"], stem, modela_image)
                if config.enable_small_image_denoise:
                    modelb_image = _remove_small_dark_components(modelb_image)
                modelb_path = _save_generated_image(style_output_roots["modelb"], stem, modelb_image)
                modela_paths.append(modela_path)
                modelb_paths.append(modelb_path)
                target_paths.append(target_path)

                style_groups[style_key]["target_paths"].append(target_path)
                style_groups[style_key]["modela_paths"].append(modela_path)
                style_groups[style_key]["modelb_paths"].append(modelb_path)
                completed_sample_keys.add(sample["sample_key"])

                panel_path = _save_compare_panel(style_output_roots["compare"], stem, target_pil, modela_image, modelb_image)
                preview_gallery.append((str(panel_path), f"{sample_index:04d}: {sample['category']} / {sample['style']} / {sample['character']}"))

                # 更新已处理计数并展示进度与 ETA
                processed_count += 1
                active_processed_count += 1
                elapsed = time.perf_counter() - active_start_time
                rate = (active_processed_count / elapsed) if elapsed > 0 else 0.0
                remaining = max(0, total_pairs - resumed_sample_count - active_processed_count)
                eta = (remaining / rate) if rate > 0 else None
                eta_str = _format_time(eta)

                progress(
                    (resumed_sample_count + active_processed_count) / total_pairs,
                    desc=f"已完成 {resumed_sample_count + active_processed_count}/{total_pairs} 个样本（续跑新增 {active_processed_count}），速率 {rate:.2f} samples/s，ETA: {eta_str}",
                )

                # 控制台周期性打印详细信息
                now = time.perf_counter()
                if now - last_log_time >= LOG_INTERVAL_SECONDS:
                    print("[batch_webui]", time.strftime("%Y-%m-%d %H:%M:%S"), "进度报告:")
                    print(f"  处理: {resumed_sample_count + active_processed_count}/{total_pairs} 样本(续跑新增 {active_processed_count}), 速率: {rate:.2f} samples/s, 已耗: {_format_time(elapsed)}, 预计剩余: {eta_str}")
                    print(f"  类别计数: {json.dumps(category_counter, ensure_ascii=False)}")
                    if skipped_samples:
                        print(f"  已跳过: {len(skipped_samples)} (示例: {skipped_samples[:3]})")
                    last_log_time = now

        elapsed_seconds = time.perf_counter() - start_time
        average_batch_seconds = float(np.mean(batch_durations)) if batch_durations else 0.0
        peak_gpu_memory_bytes = None
        if resolved_device.startswith("cuda"):
            peak_gpu_memory_bytes = float(torch.cuda.max_memory_allocated())

        metrics = _normalize_metrics(config.metrics)
        eval_summary = {}
        csv_path = None
        style_reports: List[dict] = []
        fid_reports: List[dict] = []
        fid_report_path = None
        if config.compute_metrics:
            evaluator = BatchEvaluator(device=resolved_device)
            evaluation_pairs = _collect_evaluation_pairs(processed_samples, modela_paths, modelb_paths)
            target_arrays, modela_arrays, modelb_arrays = _build_evaluation_arrays(evaluation_pairs)
            eval_summary = {
                "modela": _evaluate_model_outputs(evaluator, modela_arrays, target_arrays, metrics),
                "modelb": _evaluate_model_outputs(evaluator, modelb_arrays, target_arrays, metrics),
            }

            for style_group in style_groups.values():
                style_result = _evaluate_style_group(evaluator, style_group, metrics + ["FID"])
                style_reports.append(style_result)
                fid_reports.extend(style_result.get("fid_reports", []))

            csv_path = run_root / "summary.csv"
            _write_summary_csv(
                summary_path=csv_path,
                results=eval_summary,
                metadata={
                    "device": device_message,
                    "resolved_device": resolved_device,
                    "sample_count": total_pairs,
                    "valid_sample_count_a": len(modela_paths),
                    "valid_sample_count_b": len(modelb_paths),
                    "evaluation_pair_count": len(evaluation_pairs),
                    "elapsed_seconds": f"{elapsed_seconds:.6f}",
                    "ckpt_dir_a": config.ckpt_dir_a,
                    "ckpt_dir_b": config.ckpt_dir_b,
                    "val_root": config.val_root,
                    "source_chars_root": config.source_chars_root,
                    "note": note or "",
                },
            )

            style_report_path = run_root / "style_report.csv"
            _write_style_report_csv(style_report_path, style_reports)

            # 生成分类级别（category）汇总：将同一分类下所有风格的样本合并后再做统计
            category_groups: dict[str, dict] = {}
            for style_group in style_groups.values():
                cat = style_group.get("category")
                if cat not in category_groups:
                    category_groups[cat] = {
                        "target_paths": [],
                        "modela_paths": [],
                        "modelb_paths": [],
                        "styles": [],
                    }
                category_groups[cat]["target_paths"].extend(style_group.get("target_paths", []))
                category_groups[cat]["modela_paths"].extend(style_group.get("modela_paths", []))
                category_groups[cat]["modelb_paths"].extend(style_group.get("modelb_paths", []))
                category_groups[cat]["styles"].append(style_group.get("style"))

            category_reports: List[dict] = []
            for cat, grp in category_groups.items():
                target_paths_cat = grp.get("target_paths", [])
                modela_paths_cat = grp.get("modela_paths", [])
                modelb_paths_cat = grp.get("modelb_paths", [])

                target_arrays_cat = [_load_image(Path(p)) for p in target_paths_cat]
                modela_arrays_cat = [_load_image(Path(p)) for p in modela_paths_cat]
                modelb_arrays_cat = [_load_image(Path(p)) for p in modelb_paths_cat]

                cat_result = {
                    "category": cat,
                    "modela": _evaluate_model_outputs(evaluator, modela_arrays_cat, target_arrays_cat, metrics),
                    "modelb": _evaluate_model_outputs(evaluator, modelb_arrays_cat, target_arrays_cat, metrics),
                }

                # 计算分类级 FID（如果有样本且支持 FID）
                try:
                    if target_paths_cat and modela_paths_cat:
                        tmp_base = run_root / "category_fid_tmp" / _safe_name(cat)
                        tmp_modela = tmp_base / "modela"
                        tmp_modelb = tmp_base / "modelb"
                        tmp_target = tmp_base / "target"
                        for p in (tmp_modela, tmp_modelb, tmp_target):
                            p.mkdir(parents=True, exist_ok=True)

                        for i, p in enumerate(modela_paths_cat):
                            shutil.copy(str(p), str(tmp_modela / f"{i:06d}{Path(p).suffix}"))
                        for i, p in enumerate(modelb_paths_cat):
                            shutil.copy(str(p), str(tmp_modelb / f"{i:06d}{Path(p).suffix}"))
                        for i, p in enumerate(target_paths_cat):
                            shutil.copy(str(p), str(tmp_target / f"{i:06d}{Path(p).suffix}"))

                        # 仅在评估指标包含 FID 时尝试计算（style 层使用 metrics+['FID']）
                        try:
                            fid_value_a = float(evaluator.calculate_fid(tmp_modela, tmp_target))
                            cat_result["modela"]["FID"] = {
                                "mean": fid_value_a,
                                "std": 0.0,
                                "min": fid_value_a,
                                "max": fid_value_a,
                            }
                            fid_reports.append({
                                "level": "category",
                                "category": cat,
                                "style": "",
                                "model": "modela",
                                "fid": fid_value_a,
                                "source_dir": str(tmp_modela),
                                "target_dir": str(tmp_target),
                                "status": "ok",
                                "error": "",
                            })
                        except Exception as e:
                            cat_result.setdefault("errors", {})["modela_fid"] = str(e)
                            fid_reports.append({
                                "level": "category",
                                "category": cat,
                                "style": "",
                                "model": "modela",
                                "fid": "",
                                "source_dir": str(tmp_modela),
                                "target_dir": str(tmp_target),
                                "status": "error",
                                "error": str(e),
                            })

                        try:
                            fid_value_b = float(evaluator.calculate_fid(tmp_modelb, tmp_target))
                            cat_result["modelb"]["FID"] = {
                                "mean": fid_value_b,
                                "std": 0.0,
                                "min": fid_value_b,
                                "max": fid_value_b,
                            }
                            fid_reports.append({
                                "level": "category",
                                "category": cat,
                                "style": "",
                                "model": "modelb",
                                "fid": fid_value_b,
                                "source_dir": str(tmp_modelb),
                                "target_dir": str(tmp_target),
                                "status": "ok",
                                "error": "",
                            })
                        except Exception as e:
                            cat_result.setdefault("errors", {})["modelb_fid"] = str(e)
                            fid_reports.append({
                                "level": "category",
                                "category": cat,
                                "style": "",
                                "model": "modelb",
                                "fid": "",
                                "source_dir": str(tmp_modelb),
                                "target_dir": str(tmp_target),
                                "status": "error",
                                "error": str(e),
                            })

                except Exception as e:
                    cat_result.setdefault("errors", {})["category_aggregation"] = str(e)

                category_reports.append(cat_result)

            # 写入分类级汇总 CSV
            category_report_path = run_root / "category_report.csv"
            _write_category_report_csv(category_report_path, category_reports)

            fid_report_path = run_root / "fid_report.csv"
            _write_fid_report_csv(fid_report_path, fid_reports)

            tmp_fid_root = run_root / "category_fid_tmp"
            if tmp_fid_root.exists():
                shutil.rmtree(tmp_fid_root, ignore_errors=True)

        summary_lines = [
            "### 批量任务完成",
            f"- {device_message}",
            f"- 样本数：{total_pairs}",
            f"- 已恢复样本：{resumed_sample_count}",
            f"- 批大小：{batch_count}",
            f"- 推理耗时：{elapsed_seconds:.2f} 秒",
            f"- 平均批耗时：{average_batch_seconds:.2f} 秒",
            f"- 平均吞吐：{(active_processed_count / elapsed_seconds) if elapsed_seconds > 0 else 0.0:.2f} 样本/秒（仅统计本次新增）",
            f"- 输出目录：{run_root}",
            f"- 权重目录A：{config.ckpt_dir_a}",
            f"- 权重目录B：{config.ckpt_dir_b}",
            f"- 类别统计：{json.dumps(category_counter, ensure_ascii=False)}",
        ]
        if note:
            summary_lines.append(f"- 提示：{note}")
        if skipped_samples:
            summary_lines.append(f"- 跳过样本：{len(skipped_samples)} 个")
        if csv_path is not None:
            summary_lines.append(f"- 评价 CSV：{csv_path}")
        if style_reports:
            summary_lines.append(f"- 风格级评价 CSV：{run_root / 'style_report.csv'}")
        if fid_report_path is not None:
            summary_lines.append(f"- FID 详细 CSV：{fid_report_path}")
        for model_name in ("modela", "modelb"):
            model_metrics = eval_summary.get(model_name, {})
            if not isinstance(model_metrics, dict):
                continue
            summary_lines.append(f"- {model_name} 评价：")
            for metric_name in metrics:
                metric_stats = model_metrics.get(metric_name)
                if isinstance(metric_stats, dict):
                    summary_lines.append(
                        f"  - {metric_name}: mean={metric_stats['mean']:.6f}, std={metric_stats['std']:.6f}, "
                        f"min={metric_stats['min']:.6f}, max={metric_stats['max']:.6f}"
                    )

        if style_reports:
            summary_lines.append("- 风格级对比概览：")
            for report in style_reports[:10]:
                style_name = f"{report.get('category', '')}/{report.get('style', '')}"
                a_metrics = report.get("modela", {})
                b_metrics = report.get("modelb", {})
                a_l1 = a_metrics.get("L1", {}).get("mean", float("nan")) if isinstance(a_metrics, dict) else float("nan")
                b_l1 = b_metrics.get("L1", {}).get("mean", float("nan")) if isinstance(b_metrics, dict) else float("nan")
                summary_lines.append(f"  - {style_name}: modela L1={a_l1:.6f}, modelb L1={b_l1:.6f}")
            if len(style_reports) > 10:
                summary_lines.append(f"  - 其余 {len(style_reports) - 10} 个风格已写入 style_report.csv")

        raw_payload = {
            "device": device_message,
            "resolved_device": resolved_device,
            "sample_count": total_pairs,
            "resumed_sample_count": resumed_sample_count,
            "active_processed_count": active_processed_count,
            "elapsed_seconds": round(elapsed_seconds, 6),
            "average_batch_seconds": round(average_batch_seconds, 6),
            "samples_per_second": round((active_processed_count / elapsed_seconds) if elapsed_seconds > 0 else 0.0, 6),
            "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
            "peak_gpu_memory_human": _format_bytes(peak_gpu_memory_bytes) if peak_gpu_memory_bytes is not None else None,
            "output_dir": str(run_root),
            "ckpt_dir_a": config.ckpt_dir_a,
            "ckpt_dir_b": config.ckpt_dir_b,
            "val_root": config.val_root,
            "source_chars_root": config.source_chars_root,
            "note": note,
            "metrics": metrics,
            "results": eval_summary,
            "style_reports": style_reports,
            "fid_reports": fid_reports,
            "modela_files": [str(path) for path in modela_paths],
            "modelb_files": [str(path) for path in modelb_paths],
            "target_files": [str(path) for path in target_paths],
            "skipped_samples": skipped_samples,
            "fid_report_path": str(fid_report_path) if fid_report_path else None,
        }

        category_csv_path = None
        if style_reports:
            # category_report.csv was written when compute_metrics is True
            try:
                category_csv_path = str(category_report_path) if 'category_report_path' in locals() else None
            except Exception:
                category_csv_path = None

        perf_lines = [
            "### 性能统计",
            f"- 总耗时：{_format_time(elapsed_seconds)}",
            f"- 平均批耗时：{average_batch_seconds:.2f} 秒",
            f"- 平均吞吐：{(active_processed_count / elapsed_seconds) if elapsed_seconds > 0 else 0.0:.2f} 样本/秒（仅统计本次新增）",
        ]
        if peak_gpu_memory_bytes is not None:
            perf_lines.append(f"- 峰值显存：{_format_bytes(peak_gpu_memory_bytes)}")
        perf_lines.append(f"- 批大小：{batch_count}")
        perf_lines.append(f"- 批数量：{((total_pairs + batch_count - 1) // batch_count)}")

        return "\n".join(summary_lines), "\n".join(perf_lines), json.dumps(raw_payload, ensure_ascii=False, indent=2), preview_gallery[:60], str(run_root), str(csv_path) if csv_path else None, category_csv_path


def build_batch_webui(base_args):
    _configure_torch_runtime()
    runner = BatchInferenceRunner(base_args=base_args)

    default_ckpt_dir = getattr(base_args, "ckpt_dir", None) or "ckpt/origin"
    default_ckpt_dir_b = "ckpt/canny-44W"
    default_output_root = str(Path("outputs") / "batch_webui")
    default_val_root = str(Path("val"))
    default_source_chars_root = str(Path("."))
    default_metrics = list(EVAL_METRICS)

    with gr.Blocks(theme=gr.themes.Soft(), css=_load_css_or_empty()) as demo:
        gr.Markdown(
            "<div id='webui-hero'>"
            "<h1>fontdiff batch webui</h1>"
            "<p>目录级批量推理与批量评价入口。核心目标是一次性吃满 GPU，按批生成、按批保存、按批统计。"
            "当前版本先保持主工程不变，只额外增加独立批处理界面。</p>"
            "</div>"
        )

        with gr.Group(elem_classes="section-card"):
            gr.Markdown("### 批量任务配置")
            with gr.Row():
                val_root = gr.Textbox(value=default_val_root, label="val 根目录", placeholder="例如: E:/Projects/fontdiff/val")
                source_chars_root = gr.Textbox(value=default_source_chars_root, label="Source Characters txt 根目录", placeholder="例如: E:/Projects/fontdiff")
            with gr.Row():
                resume_run_root = gr.Textbox(value="", label="续跑目录（可选）", placeholder="例如: E:/Projects/fontdiff/outputs/batch_webui/20260529_120000")
                output_root = gr.Textbox(value=default_output_root, label="输出根目录")
                ckpt_dir_a = gr.Textbox(value=default_ckpt_dir, label="Model A CKPT 目录")
                ckpt_dir_b = gr.Textbox(value=default_ckpt_dir_b, label="Model B CKPT 目录")
                ttf_path = gr.Textbox(value=str(getattr(base_args, "ttf_path", None) or "ttf/KaiXinSongA.ttf"), label="TTF 路径")
            with gr.Row():
                device_mode = gr.Dropdown(choices=["auto", "cpu", "cuda:0"], value=base_args.device, label="运行设备")
                batch_size = gr.Slider(1, 128, value=8, step=1, label="批大小")
            with gr.Row():
                num_inference_steps = gr.Slider(10, 100, value=int(getattr(base_args, "num_inference_steps", 20)), step=1, label="采样步数")
                guidance_scale = gr.Slider(1.0, 12.0, value=float(getattr(base_args, "guidance_scale", 7.5)), step=0.5, label="Guidance Scale")
                save_panels = gr.Checkbox(value=True, label="保存对比面板")
                compute_metrics = gr.Checkbox(value=True, label="任务完成后统一评价")
                denoise_small_image = gr.Checkbox(value=True, label="启用小图去噪（仅 modelb，<=64px）")

            metrics = gr.CheckboxGroup(choices=EVAL_METRICS, value=default_metrics, label="评价指标")
            run_button = gr.Button("开始批量推理", variant="primary")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                summary = gr.Markdown(value="请先填写目录和参数，然后开始任务。")
                perf_stats = gr.Markdown(value="### 性能统计\n- 尚未运行")
                preview_gallery = gr.Gallery(label="结果预览", columns=2, height=420)
            with gr.Column(scale=1):
                raw_json = gr.Textbox(label="原始结果", lines=18, max_lines=24, interactive=False, show_copy_button=True)
                run_dir = gr.Textbox(label="输出目录", interactive=False)
                csv_file = gr.Textbox(label="评价 CSV", interactive=False)
                category_csv = gr.Textbox(label="分类汇总 CSV", interactive=False)

        def _submit_batch(
            val_root_value,
            source_chars_root_value,
            resume_run_root_value,
            output_root_value,
            ckpt_dir_a_value,
            ckpt_dir_b_value,
            ttf_path_value,
            device_value,
            batch_size_value,
            steps_value,
            guidance_value,
            save_panels_value,
            compute_metrics_value,
                denoise_value,
            metrics_value,
        ):
            config = BatchWebUIConfig(
                val_root=val_root_value,
                source_chars_root=source_chars_root_value,
                ttf_path=ttf_path_value,
                ckpt_dir_a=ckpt_dir_a_value,
                ckpt_dir_b=ckpt_dir_b_value,
                device_mode=device_value,
                resume_run_root=resume_run_root_value,
                output_root=output_root_value,
                batch_size=int(batch_size_value),
                num_inference_steps=int(steps_value),
                guidance_scale=float(guidance_value),
                compute_metrics=bool(compute_metrics_value),
                save_panels=bool(save_panels_value),
                enable_small_image_denoise=bool(denoise_value),
                metrics=_normalize_metrics(metrics_value),
            )
            return runner.run(config)

        run_button.click(
            fn=_submit_batch,
            inputs=[
                val_root,
                source_chars_root,
                resume_run_root,
                output_root,
                ckpt_dir_a,
                ckpt_dir_b,
                ttf_path,
                device_mode,
                batch_size,
                num_inference_steps,
                guidance_scale,
                save_panels,
                compute_metrics,
                denoise_small_image,
                metrics,
            ],
            outputs=[summary, perf_stats, raw_json, preview_gallery, run_dir, csv_file, category_csv],
            api_name=False,
        )

    return demo


if __name__ == "__main__":
    _configure_torch_runtime()
    base_args = arg_parse()
    base_args.demo = True
    base_args.ckpt_dir = getattr(base_args, "ckpt_dir", None) or "ckpt/origin"
    base_args.ttf_path = getattr(base_args, "ttf_path", None) or "ttf/KaiXinSongA.ttf"
    demo = build_batch_webui(base_args)
    demo.queue(default_concurrency_limit=1)
    try:
        demo.launch(debug=True, show_api=False, share=False, server_name="0.0.0.0", inbrowser=True)
    except ValueError as launch_error:
        if "localhost is not accessible" in str(launch_error):
            print("Localhost is not accessible. Retrying with share link enabled.")
            demo.launch(debug=True, show_api=False, share=True, inbrowser=True)
        else:
            raise