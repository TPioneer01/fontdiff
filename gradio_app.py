import copy
import csv
import json
import os
import random
import shutil
import tempfile
import time
from functools import lru_cache
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from evaluation import BatchEvaluator, SingleSampleEvaluator
from sample import (
    arg_parse,
    sampling,
    load_fontdiffuer_pipeline,
    resolve_runtime_device,
)
from utils import load_ttf, ttf2im


COMPARE_PIPE_CACHE = {
    "key": None,
    "pipe_a": None,
    "pipe_b": None,
    "device": None,
    "message": None,
}


def _parse_source_characters(source_characters):
    if source_characters is None:
        return []

    characters = []
    for line in str(source_characters).replace("，", ",").splitlines():
        for character in line.split(","):
            character = character.strip()
            if character:
                characters.append(character)
    return characters


def _safe_dirname(text):
    cleaned = "".join(character if character.isalnum() else "_" for character in str(text))
    cleaned = cleaned.strip("_")
    return cleaned or "char"


def _safe_filename_stem(text):
    stem = str(text).strip()
    invalid_chars = set('<>:"/\\|?*')
    if not stem or any(character in invalid_chars for character in stem):
        return _safe_dirname(stem)
    return stem


@lru_cache(maxsize=8)
def _cached_font(ttf_path, fsize):
    return load_ttf(ttf_path=ttf_path, fsize=fsize)


def _render_character_image(ttf_path, character, fsize=128):
    font = _cached_font(ttf_path, fsize)
    image = ttf2im(font=font, char=character, fsize=fsize)
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


def _build_compare_panel(content_image, style_image, generated_image, resolution):
    content_image = _ensure_rgb_pil_image(content_image, resolution)
    style_image = _ensure_rgb_pil_image(style_image, resolution)
    generated_image = _ensure_rgb_pil_image(generated_image, resolution)

    canvas = Image.new("RGB", (resolution * 3, resolution), color=(255, 255, 255))
    canvas.paste(content_image, (0, 0))
    canvas.paste(style_image, (resolution, 0))
    canvas.paste(generated_image, (resolution * 2, 0))
    return canvas


def _build_ab_compare_panel(image_a, image_b, resolution=None):
    image_a = _ensure_rgb_pil_image(image_a, resolution)
    image_b = _ensure_rgb_pil_image(image_b, resolution)
    if resolution is None:
        resolution = image_a.size[0]
    canvas = Image.new("RGB", (resolution * 2, resolution), color=(255, 255, 255))
    canvas.paste(image_a, (0, 0))
    canvas.paste(image_b, (resolution, 0))
    return canvas


def _resolve_output_resolution(base_args, generated_image, content_image, style_image):
    for candidate in (getattr(base_args, "resolution", None), getattr(generated_image, "size", [None])[0]):
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    if isinstance(content_image, Image.Image):
        return content_image.size[0]
    if isinstance(style_image, Image.Image):
        return style_image.size[0]
    return 128


def _save_single_model_output(save_dir, filename_stem, generated_image, base_args, model_tag):
    os.makedirs(save_dir, exist_ok=True)

    resolution = _resolve_output_resolution(base_args, generated_image, None, None)
    single_image = _ensure_rgb_pil_image(generated_image, resolution)

    single_path = Path(save_dir) / f"{model_tag}_{filename_stem}.png"

    single_image.save(single_path)
    return str(single_path)


def _save_ab_compare_output(save_dir, filename_stem, image_a, image_b, base_args):
    os.makedirs(save_dir, exist_ok=True)

    resolution = _resolve_output_resolution(base_args, image_a, image_a, image_b)
    compare_image = _build_ab_compare_panel(image_a, image_b, resolution=resolution)
    compare_path = Path(save_dir) / f"compare_{filename_stem}.png"
    compare_image.save(compare_path, quality=95)
    return str(compare_path)


def _clone_args_for_ckpt(base_args, ckpt_dir, device):
    cloned_args = copy.deepcopy(base_args)
    cloned_args.ckpt_dir = ckpt_dir
    cloned_args.device = device
    return cloned_args


def _load_compare_pipes(base_args, device_mode, ckpt_dir_a, ckpt_dir_b):
    resolved_device, device_message = resolve_runtime_device(device_mode)
    cache_key = (resolved_device, ckpt_dir_a, ckpt_dir_b)

    if COMPARE_PIPE_CACHE["key"] == cache_key:
        return (
            COMPARE_PIPE_CACHE["pipe_a"],
            COMPARE_PIPE_CACHE["pipe_b"],
            resolved_device,
            device_message,
        )

    args_a = _clone_args_for_ckpt(base_args, ckpt_dir_a, resolved_device)
    args_b = _clone_args_for_ckpt(base_args, ckpt_dir_b, resolved_device)

    pipe_a = load_fontdiffuer_pipeline(args=args_a)
    pipe_b = load_fontdiffuer_pipeline(args=args_b)

    COMPARE_PIPE_CACHE["key"] = cache_key
    COMPARE_PIPE_CACHE["pipe_a"] = pipe_a
    COMPARE_PIPE_CACHE["pipe_b"] = pipe_b
    COMPARE_PIPE_CACHE["device"] = resolved_device
    COMPARE_PIPE_CACHE["message"] = device_message

    return pipe_a, pipe_b, resolved_device, device_message


def _prepare_run_args(base_args, ckpt_dir, resolved_device, sampling_step, guidance_scale, save_image_dir):
    run_args = copy.deepcopy(base_args)
    run_args.ckpt_dir = ckpt_dir
    run_args.device = resolved_device
    run_args.num_inference_steps = int(sampling_step)
    run_args.guidance_scale = guidance_scale
    run_args.save_image = False
    run_args.save_image_dir = save_image_dir
    return run_args


EVAL_METRICS = ["L1", "L2", "RMSE", "PSNR", "SSIM", "LPIPS"]


def _normalize_metric_selection(selected_metrics):
    metrics = selected_metrics or EVAL_METRICS
    normalized = []
    for metric in metrics:
        metric_name = str(metric).strip().upper()
        if metric_name in EVAL_METRICS and metric_name not in normalized:
            normalized.append(metric_name)
    return normalized or list(EVAL_METRICS)


def _resolve_upload_path(file_item):
    if file_item is None:
        return None
    if isinstance(file_item, (str, os.PathLike)):
        return str(file_item)
    for attribute in ("path", "name"):
        value = getattr(file_item, attribute, None)
        if value:
            return str(value)
    return str(file_item)


def _normalize_file_list(files):
    if files is None:
        return []
    if isinstance(files, (str, os.PathLike)):
        return [str(files)]
    normalized = []
    for file_item in files:
        resolved = _resolve_upload_path(file_item)
        if resolved:
            normalized.append(resolved)
    return sorted(normalized, key=lambda path: os.path.basename(path).lower())


def _load_image_as_array(image_input):
    if image_input is None:
        return None
    if isinstance(image_input, np.ndarray):
        image_array = image_input
    elif isinstance(image_input, (str, os.PathLike)):
        with Image.open(image_input) as image_file:
            image_array = np.array(image_file.convert("RGB"))
    elif hasattr(image_input, "convert"):
        image_array = np.array(image_input.convert("RGB"))
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)!r}")

    if image_array.ndim == 3 and image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    return image_array


def _format_metric_value(value):
    if isinstance(value, (float, int)) and not np.isfinite(value):
        return "∞"
    if isinstance(value, (float, int)):
        return f"{float(value):.6f}"
    return str(value)


def _build_single_result_markdown(results, elapsed_seconds, device_message):
    lines = [
        "### 单样本评价结果",
        f"- {device_message}",
        f"- 耗时：{elapsed_seconds:.2f} 秒",
    ]
    for metric_name in EVAL_METRICS:
        if metric_name in results:
            lines.append(f"- {metric_name}: {_format_metric_value(results[metric_name])}")
    return "\n".join(lines)


def _build_batch_result_markdown(results, elapsed_seconds, device_message, sample_count, fid_value=None, note=None):
    lines = [
        "### 批量评价结果",
        f"- {device_message}",
        f"- 样本数：{sample_count}",
        f"- 耗时：{elapsed_seconds:.2f} 秒",
    ]
    if note:
        lines.append(f"- 提示：{note}")
    for metric_name in EVAL_METRICS:
        metric_result = results.get(metric_name)
        if not isinstance(metric_result, dict):
            continue
        lines.append(
            f"- {metric_name}: mean={_format_metric_value(metric_result['mean'])}, "
            f"std={_format_metric_value(metric_result['std'])}, "
            f"min={_format_metric_value(metric_result['min'])}, "
            f"max={_format_metric_value(metric_result['max'])}"
        )
    if fid_value is not None:
        lines.append(f"- FID: {_format_metric_value(fid_value)}")
    return "\n".join(lines)


def _build_batch_result_copy_text(results, elapsed_seconds, device_message, sample_count, fid_value=None, note=None):
    payload = {
        "device": device_message,
        "sample_count": sample_count,
        "elapsed_seconds": round(elapsed_seconds, 6),
        "note": note,
        "results": results,
    }
    if fid_value is not None:
        payload["FID"] = fid_value
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _write_batch_results_csv(results, device_message, sample_count, elapsed_seconds, note=None, fid_value=None):
    output_dir = Path("outputs") / "evaluation_reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"batch_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    rows = [
        ["metric", "mean", "std", "min", "max"],
    ]
    for metric_name in EVAL_METRICS:
        metric_result = results.get(metric_name)
        if not isinstance(metric_result, dict):
            continue
        rows.append([
            metric_name,
            metric_result.get("mean", ""),
            metric_result.get("std", ""),
            metric_result.get("min", ""),
            metric_result.get("max", ""),
        ])

    if fid_value is not None:
        rows.append(["FID", fid_value, 0.0, fid_value, fid_value])

    with csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["summary", "value"])
        writer.writerow(["device", device_message])
        writer.writerow(["sample_count", sample_count])
        writer.writerow(["elapsed_seconds", f"{elapsed_seconds:.6f}"])
        if note:
            writer.writerow(["note", note])
        writer.writerow([])
        for row in rows:
            writer.writerow(row)

    return str(csv_path)


def _copy_files_to_temp_dir(file_paths, prefix):
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    for index, file_path in enumerate(file_paths):
        source_path = Path(file_path)
        suffix = source_path.suffix or ".png"
        target_path = Path(temp_dir) / f"{index:04d}_{source_path.stem}{suffix}"
        shutil.copy2(str(source_path), str(target_path))
    return temp_dir


def evaluate_single_sample_ui(generated_image, reference_image, selected_metrics, device_mode):
    if generated_image is None or reference_image is None:
        return "### 请先上传生成图和参考图。", {}

    resolved_device, device_message = resolve_runtime_device(device_mode)
    evaluator = SingleSampleEvaluator(device=resolved_device)
    metrics = _normalize_metric_selection(selected_metrics)

    try:
        generated_array = _load_image_as_array(generated_image)
        reference_array = _load_image_as_array(reference_image)
        start_time = time.perf_counter()
        results = evaluator.evaluate(generated_array, reference_array, metrics=metrics)
        elapsed_seconds = time.perf_counter() - start_time
        summary = _build_single_result_markdown(results, elapsed_seconds, device_message)
        return summary, results
    except Exception as error:
        return f"### 单样本评价失败\n- {error}", {}


def evaluate_batch_ui(generated_files, reference_files, selected_metrics, device_mode, compute_fid):
    generated_paths = _normalize_file_list(generated_files)
    reference_paths = _normalize_file_list(reference_files)

    if not generated_paths or not reference_paths:
        return "### 请先上传两组图像。", "{}", None

    resolved_device, device_message = resolve_runtime_device(device_mode)
    evaluator = BatchEvaluator(device=resolved_device)
    metrics = _normalize_metric_selection(selected_metrics)

    generated_pairs = []
    reference_pairs = []
    paired_count = min(len(generated_paths), len(reference_paths))
    note = None
    if len(generated_paths) != len(reference_paths):
        note = f"上传数量不一致，当前仅按排序后前 {paired_count} 对进行评价。"

    if compute_fid and paired_count < 2:
        compute_fid = False
        note = f"{note} FID 需要至少 2 个样本，当前已自动跳过。" if note else "FID 需要至少 2 个样本，当前已自动跳过。"

    try:
        for index in range(paired_count):
            generated_pairs.append(_load_image_as_array(generated_paths[index]))
            reference_pairs.append(_load_image_as_array(reference_paths[index]))

        start_time = time.perf_counter()
        results = evaluator.evaluate_batch(generated_pairs, reference_pairs, metrics=metrics)

        fid_value = None
        temp_dirs = []
        if compute_fid:
            fake_dir = _copy_files_to_temp_dir(generated_paths[:paired_count], "fontdiff_fake_")
            real_dir = _copy_files_to_temp_dir(reference_paths[:paired_count], "fontdiff_real_")
            temp_dirs.extend([fake_dir, real_dir])
            try:
                fid_value = evaluator.calculate_fid(fake_dir, real_dir)
                results["FID"] = {
                    "mean": fid_value,
                    "std": 0.0,
                    "min": fid_value,
                    "max": fid_value,
                }
            finally:
                for temp_dir in temp_dirs:
                    shutil.rmtree(temp_dir, ignore_errors=True)

        elapsed_seconds = time.perf_counter() - start_time
        summary = _build_batch_result_markdown(
            results=results,
            elapsed_seconds=elapsed_seconds,
            device_message=device_message,
            sample_count=paired_count,
            fid_value=fid_value,
            note=note,
        )
        raw_results_text = _build_batch_result_copy_text(
            results=results,
            elapsed_seconds=elapsed_seconds,
            device_message=device_message,
            sample_count=paired_count,
            fid_value=fid_value,
            note=note,
        )
        csv_path = _write_batch_results_csv(
            results=results,
            device_message=device_message,
            sample_count=paired_count,
            elapsed_seconds=elapsed_seconds,
            note=note,
            fid_value=fid_value,
        )
        return summary, raw_results_text, csv_path
    except Exception as error:
        return f"### 批量评价失败\n- {error}", json.dumps({"error": str(error)}, ensure_ascii=False, indent=2), None


def _run_single_model(base_args,
                      pipe,
                      resolved_device,
                      ckpt_dir,
                      source_image,
                      source_characters_list,
                      reference_image,
                      sampling_step,
                      guidance_scale,
                      model_tag,
                      run_root):
    if source_image is not None:
        save_dir = run_root
        filename_stem = _safe_filename_stem("source_image")

        run_args = _prepare_run_args(
            base_args=base_args,
            ckpt_dir=ckpt_dir,
            resolved_device=resolved_device,
            sampling_step=sampling_step,
            guidance_scale=guidance_scale,
            save_image_dir=save_dir,
        )
        run_args.character_input = False
        run_args.content_character = None
        run_args.seed = random.randint(0, 10000)

        generated_image = sampling(
            args=run_args,
            pipe=pipe,
            content_image=source_image,
            style_image=reference_image,
        )
        if generated_image is None:
            return [], [f"{model_tag}: source_image -> skipped."]

        single_path = _save_single_model_output(
            save_dir=save_dir,
            filename_stem=filename_stem,
            generated_image=generated_image,
            base_args=base_args,
            model_tag=model_tag,
        )
        return [(generated_image, "source_image")], [f"{model_tag}: saved {single_path}"]

    results = []
    status_lines = []

    for index, character in enumerate(source_characters_list, start=1):
        save_dir = run_root
        filename_stem = _safe_filename_stem(character)

        run_args = _prepare_run_args(
            base_args=base_args,
            ckpt_dir=ckpt_dir,
            resolved_device=resolved_device,
            sampling_step=sampling_step,
            guidance_scale=guidance_scale,
            save_image_dir=save_dir,
        )
        run_args.character_input = True
        run_args.content_character = character
        run_args.seed = random.randint(0, 10000)

        generated_image = sampling(
            args=run_args,
            pipe=pipe,
            content_image=None,
            style_image=reference_image,
        )
        if generated_image is not None:
            results.append((generated_image, character))
            single_path = _save_single_model_output(
                save_dir=save_dir,
                filename_stem=filename_stem,
                generated_image=generated_image,
                base_args=base_args,
                model_tag=model_tag,
            )
            status_lines.append(f"{model_tag}: {index:02d}. {character} -> {single_path}")
        else:
            status_lines.append(f"{model_tag}: {index:02d}. {character} -> skipped (character not in font).")

    status_lines.append(f"{model_tag}: saved {len(results)} result(s) under {run_root}.")
    return results, status_lines


def _run_character_models_interleaved(base_args,
                                      pipe_a,
                                      pipe_b,
                                      resolved_device,
                                      ckpt_dir_a,
                                      ckpt_dir_b,
                                      source_characters_list,
                                      reference_image,
                                      sampling_step,
                                      guidance_scale,
                                      run_root,
                                      compare_root,
                                      progress=None):
    results_a = []
    results_b = []
    status_a = []
    status_b = []
    pair_status_lines = []
    total_characters = len(source_characters_list)

    for index, character in enumerate(source_characters_list, start=1):
        character_progress_base = (index - 1) / total_characters if total_characters > 0 else 0
        character_progress_step = 1 / total_characters if total_characters > 0 else 0

        if progress is not None and total_characters > 0:
            progress(
                character_progress_base,
                desc=f"正在推理第 {index}/{total_characters} 个字：{character}（模型 A）",
            )

        char_outputs = {}
        for model_tag, pipe, ckpt_dir, status_lines, results in (
            ("model_a", pipe_a, ckpt_dir_a, status_a, results_a),
            ("model_b", pipe_b, ckpt_dir_b, status_b, results_b),
        ):
            save_dir = run_root
            filename_stem = _safe_filename_stem(character)

            run_args = _prepare_run_args(
                base_args=base_args,
                ckpt_dir=ckpt_dir,
                resolved_device=resolved_device,
                sampling_step=sampling_step,
                guidance_scale=guidance_scale,
                save_image_dir=save_dir,
            )
            run_args.character_input = True
            run_args.content_character = character
            run_args.seed = random.randint(0, 10000)

            generated_image = sampling(
                args=run_args,
                pipe=pipe,
                content_image=None,
                style_image=reference_image,
            )
            if generated_image is not None:
                results.append((generated_image, character))
                char_outputs[model_tag] = generated_image
                single_path = _save_single_model_output(
                    save_dir=save_dir,
                    filename_stem=filename_stem,
                    generated_image=generated_image,
                    base_args=base_args,
                    model_tag=model_tag,
                )
                status_lines.append(f"{model_tag}: {index:02d}. {character} -> {single_path}")
            else:
                status_lines.append(f"{model_tag}: {index:02d}. {character} -> skipped (character not in font).")

            if progress is not None and total_characters > 0:
                stage_offset = 0.45 if model_tag == "model_a" else 0.85
                progress(
                    character_progress_base + character_progress_step * stage_offset,
                    desc=f"正在推理第 {index}/{total_characters} 个字：{character}（{model_tag} 完成）",
                )

        image_a = char_outputs.get("model_a")
        image_b = char_outputs.get("model_b")
        if image_a is None or image_b is None:
            pair_status_lines.append(f"pair: {index:02d}. {character} -> skipped (missing output from one model).")
            if progress is not None and total_characters > 0:
                progress(
                    index / total_characters,
                    desc=f"正在推理第 {index}/{total_characters} 个字：{character}（跳过比对）",
                )
            continue

        compare_path = _save_ab_compare_output(
            save_dir=compare_root,
            filename_stem=_safe_filename_stem(character),
            image_a=image_a,
            image_b=image_b,
            base_args=base_args,
        )
        pair_status_lines.append(f"pair: {index:02d}. {character} -> {compare_path}")

        if progress is not None and total_characters > 0:
            progress(
                index / total_characters,
                desc=f"正在推理第 {index}/{total_characters} 个字：{character}（比对完成）",
            )

    if progress is not None and total_characters > 0:
        progress(1, desc=f"推理完成，共 {total_characters} 个字")

    return results_a, results_b, status_a, status_b, pair_status_lines


def run_compare_fontdiffuer(source_image,
                            source_characters,
                            reference_image,
                            sampling_step,
                            guidance_scale,
                            device_mode,
                            ckpt_dir_a,
                            ckpt_dir_b,
                            progress=gr.Progress()):
    source_characters_list = _parse_source_characters(source_characters)

    if source_image is None and not source_characters_list:
        return [], [], "Please provide a source image or at least one source character.", "", ""

    compare_root = os.path.join("outputs", "gradio_compare", datetime.now().strftime("%Y%m%d_%H%M%S"))

    pipe_a, pipe_b, resolved_device, device_message = _load_compare_pipes(
        base_args=ui_args,
        device_mode=device_mode,
        ckpt_dir_a=ckpt_dir_a,
        ckpt_dir_b=ckpt_dir_b,
    )

    run_root = os.path.join("outputs", "gradio_compare", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_root, exist_ok=True)

    if source_image is not None:
        results_a, status_a = _run_single_model(
            base_args=ui_args,
            pipe=pipe_a,
            resolved_device=resolved_device,
            ckpt_dir=ckpt_dir_a,
            source_image=source_image,
            source_characters_list=source_characters_list,
            reference_image=reference_image,
            sampling_step=sampling_step,
            guidance_scale=guidance_scale,
            model_tag="model_a",
            run_root=run_root,
        )
        results_b, status_b = _run_single_model(
            base_args=ui_args,
            pipe=pipe_b,
            resolved_device=resolved_device,
            ckpt_dir=ckpt_dir_b,
            source_image=source_image,
            source_characters_list=source_characters_list,
            reference_image=reference_image,
            sampling_step=sampling_step,
            guidance_scale=guidance_scale,
            model_tag="model_b",
            run_root=run_root,
        )

        labels = ["source_image"]
        image_map_a = {label: image for image, label in results_a}
        image_map_b = {label: image for image, label in results_b}
        pair_status_lines = []

        for index, label in enumerate(labels, start=1):
            image_a = image_map_a.get(label)
            image_b = image_map_b.get(label)
            if image_a is None or image_b is None:
                pair_status_lines.append(f"pair: {index:02d}. {label} -> skipped (missing output from one model).")
                continue

            compare_path = _save_ab_compare_output(
                save_dir=compare_root,
                filename_stem=_safe_filename_stem(label),
                image_a=image_a,
                image_b=image_b,
                base_args=ui_args,
            )
            pair_status_lines.append(f"pair: {index:02d}. {label} -> {compare_path}")
    else:
        results_a, results_b, status_a, status_b, pair_status_lines = _run_character_models_interleaved(
            base_args=ui_args,
            pipe_a=pipe_a,
            pipe_b=pipe_b,
            resolved_device=resolved_device,
            ckpt_dir_a=ckpt_dir_a,
            ckpt_dir_b=ckpt_dir_b,
            source_characters_list=source_characters_list,
            reference_image=reference_image,
            sampling_step=sampling_step,
            guidance_scale=guidance_scale,
            run_root=run_root,
            compare_root=compare_root,
            progress=progress,
        )

    runtime_status = "\n".join([
        f"{device_message} Current runtime device: {resolved_device}.",
        f"Model A checkpoint: {ckpt_dir_a}",
        f"Model B checkpoint: {ckpt_dir_b}",
        f"Saved model outputs under {run_root}.",
        f"Saved A/B compare images under {compare_root}.",
    ] + status_a + status_b)

    if pair_status_lines:
        runtime_status = "\n".join([runtime_status] + pair_status_lines)

    return results_a, results_b, runtime_status, run_root, compare_root


WEBUI_CSS = """
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(59, 130, 246, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(16, 185, 129, 0.10), transparent 24%),
        linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
}

#webui-hero {
    text-align: center;
    margin-bottom: 0.75rem;
}

#webui-hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    margin-bottom: 0.3rem;
}

#webui-hero p {
    font-size: 1rem;
    color: #4b5563;
    margin: 0 auto;
    max-width: 760px;
}

.section-card {
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 22px;
    padding: 20px 20px 16px 20px;
    background: rgba(255, 255, 255, 0.72);
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    backdrop-filter: blur(8px);
}

.section-card h3 {
    margin-top: 0;
}

.hint-pill {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    margin: 0.15rem 0.3rem 0.15rem 0;
    border-radius: 999px;
    background: rgba(37, 99, 235, 0.08);
    color: #1d4ed8;
    font-size: 0.86rem;
    font-weight: 600;
}

.workflow-step {
    padding: 0.55rem 0.75rem;
    border-left: 3px solid #3b82f6;
    background: rgba(59, 130, 246, 0.06);
    border-radius: 0 12px 12px 0;
    margin-bottom: 0.5rem;
}

.gr-button {
    border-radius: 999px !important;
}

.gradio-container .tab-nav button {
    border-radius: 999px !important;
}

.compare-button,
.batch-run-button,
.single-run-button {
    min-height: 56px;
    font-size: 1.02rem;
    font-weight: 800;
    border-radius: 18px !important;
    box-shadow: 0 14px 30px rgba(37, 99, 235, 0.22);
}

.compare-button {
    background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 45%, #0f766e 100%) !important;
}

.compare-button:hover,
.batch-run-button:hover,
.single-run-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 34px rgba(37, 99, 235, 0.28);
}

.upload-shell,
.result-shell,
.control-shell {
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.72);
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    backdrop-filter: blur(8px);
    padding: 12px;
}

.upload-box {
    min-height: 280px;
    max-height: 280px;
    overflow-y: auto;
    padding-right: 6px;
}

.upload-box > div {
    height: 100%;
}

.result-box {
    min-height: 260px;
}

.result-json textarea {
    max-height: 240px;
    overflow-y: auto;
    background: rgba(255, 255, 255, 0.85) !important;
}

.result-json button {
    border-radius: 999px !important;
}

"""


def build_webui(ui_args):
    default_ckpt_dir_a = 'ckpt/origin'
    default_ckpt_dir_b = 'ckpt/canny-44W'
    pipe_a, pipe_b, loaded_device, loaded_message = _load_compare_pipes(
        base_args=ui_args,
        device_mode=ui_args.device,
        ckpt_dir_a=default_ckpt_dir_a,
        ckpt_dir_b=default_ckpt_dir_b,
    )
    device_status_default = f"{loaded_message} Current runtime device: {loaded_device}."

    with gr.Blocks(theme=gr.themes.Soft(), css=WEBUI_CSS) as demo:
        gr.Markdown(
            "<div id='webui-hero'>"
            "<h1>FontDiffuser Studio</h1>"
            "<p>这里同时提供模型对比推理和图像评价中心。你可以上传单张图做 L1 / L2 / RMSE / PSNR / SSIM / LPIPS 评价，"
            "也可以上传一组图批量统计并计算 FID。</p>"
            "</div>"
        )

        with gr.Group(elem_classes="section-card"):
            gr.Markdown("### 快速引导")
            gr.Markdown(
                """
                <div class='workflow-step'>1. 先选择上方标签页，决定是做模型对比还是进入评价中心。</div>
                <div class='workflow-step'>2. 单样本模式上传一对图像，批量模式上传两组图像列表。</div>
                <div class='workflow-step'>3. 勾选需要的指标，批量模式可以额外导出 CSV 并计算 FID。</div>
                <div>
                    <span class='hint-pill'>支持 GPU / CPU</span>
                    <span class='hint-pill'>支持 PNG / JPG / WEBP</span>
                    <span class='hint-pill'>FID 仅批量可用</span>
                </div>
                """
            )

        with gr.Tabs():
            with gr.Tab("模型对比"):
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("### 双模型推理对比")
                    gr.Markdown("输入一个参考图或一组 Source Character，分别用两份权重推理并并排对比结果。")

                    with gr.Row():
                        source_image = gr.Image(width=320, label='Source Image (optional)', image_mode='RGB', type='pil')
                        reference_image = gr.Image(width=320, label='Reference Image', image_mode='RGB', type='pil')

                    source_characters = gr.Textbox(
                        value='中\n国\n矿\n业\n大\n学\n',
                        lines=10,
                        max_lines=10,
                        label='Source Characters',
                        placeholder='每行一个字符，也支持用英文逗号分隔，例如: 中,国,矿,业,大,学',
                    )

                    with gr.Row():
                        device_mode = gr.Dropdown(
                            choices=["auto", "cpu", "cuda:0"],
                            value=ui_args.device,
                            label="Runtime Device",
                        )
                        ckpt_dir_a = gr.Textbox(value=default_ckpt_dir_a, label="Model A CKPT Dir")
                        ckpt_dir_b = gr.Textbox(value=default_ckpt_dir_b, label="Model B CKPT Dir")

                    with gr.Row(equal_height=True):
                        with gr.Column(scale=1):
                            sampling_step = gr.Slider(20, 50, value=20, step=10, label="Sampling Step")
                        with gr.Column(scale=1):
                            guidance_scale = gr.Slider(1, 12, value=7.5, step=0.5, label="Guidance Scale")
                        with gr.Column(scale=1):
                            run_button = gr.Button('Run Comparison', variant='primary', elem_classes='compare-button')

                    with gr.Row():
                        with gr.Column():
                            output_gallery_a = gr.Gallery(label='Model A Results', columns=2, height=360)
                            save_dir_text_a = gr.Textbox(value='', label='Model A Saved Directory', interactive=False)
                        with gr.Column():
                            output_gallery_b = gr.Gallery(label='Model B Results', columns=2, height=360)
                            save_dir_text_b = gr.Textbox(value='', label='Model B Saved Directory', interactive=False)

                    runtime_status = gr.Textbox(value=device_status_default, label='Runtime Status', interactive=False, lines=4)

                    run_button.click(
                        fn=run_compare_fontdiffuer,
                        inputs=[source_image,
                                source_characters,
                                reference_image,
                                sampling_step,
                                guidance_scale,
                                device_mode,
                                ckpt_dir_a,
                                ckpt_dir_b],
                        outputs=[output_gallery_a,
                                 output_gallery_b,
                                 runtime_status,
                                 save_dir_text_a,
                                 save_dir_text_b],
                        api_name=False)

            with gr.Tab("评价中心"):
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("### 图像质量评价中心")
                    gr.Markdown("支持单样本评价与多样本批量评价；FID 仅在批量模式下提供。")

                with gr.Tabs():
                    with gr.Tab("单样本评价"):
                        with gr.Group(elem_classes="section-card"):
                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1, elem_classes="control-shell"):
                                    single_generated_image = gr.Image(
                                        label="生成图像",
                                        image_mode='RGB',
                                        type='pil',
                                        height=320,
                                    )
                                    single_reference_image = gr.Image(
                                        label="参考图像",
                                        image_mode='RGB',
                                        type='pil',
                                        height=320,
                                    )
                                    single_eval_button = gr.Button("开始单样本评价", variant='primary', elem_classes='single-run-button')
                                with gr.Column(scale=1, elem_classes="result-shell"):
                                    single_metrics = gr.CheckboxGroup(
                                        choices=EVAL_METRICS,
                                        value=EVAL_METRICS,
                                        label="评价指标",
                                    )
                                    single_device_mode = gr.Dropdown(
                                        choices=["auto", "cpu", "cuda:0"],
                                        value=ui_args.device,
                                        label="Runtime Device",
                                    )
                                    single_summary = gr.Markdown(value="请上传两张图像，然后点击按钮开始评价。")
                                    single_raw_results = gr.JSON(label="原始结果")

                        single_eval_button.click(
                            fn=evaluate_single_sample_ui,
                            inputs=[single_generated_image,
                                    single_reference_image,
                                    single_metrics,
                                    single_device_mode],
                            outputs=[single_summary, single_raw_results],
                            api_name=False,
                        )

                    with gr.Tab("批量评价 + FID"):
                        with gr.Group(elem_classes="section-card"):
                            batch_eval_button = gr.Button("开始批量评价", variant='primary', elem_classes='batch-run-button')

                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1, elem_classes="upload-shell"):
                                    batch_generated_files = gr.File(
                                        label="生成图像列表",
                                        file_count="multiple",
                                        type="filepath",
                                        elem_classes="upload-box",
                                    )
                                with gr.Column(scale=1, elem_classes="upload-shell"):
                                    batch_reference_files = gr.File(
                                        label="参考图像列表",
                                        file_count="multiple",
                                        type="filepath",
                                        elem_classes="upload-box",
                                    )

                            with gr.Row():
                                batch_metrics = gr.CheckboxGroup(
                                    choices=EVAL_METRICS,
                                    value=EVAL_METRICS,
                                    label="评价指标",
                                )
                                batch_compute_fid = gr.Checkbox(
                                    value=True,
                                    label="同时计算 FID（仅批量模式）",
                                )
                                batch_device_mode = gr.Dropdown(
                                    choices=["auto", "cpu", "cuda:0"],
                                    value=ui_args.device,
                                    label="Runtime Device",
                                )

                            batch_csv_download = gr.DownloadButton(
                                label="下载 CSV 结果",
                                value=None,
                                visible=False,
                            )

                            with gr.Row(equal_height=True):
                                with gr.Column(scale=1, elem_classes="result-shell result-box"):
                                    batch_summary = gr.Markdown(value="请分别上传生成图像和参考图像列表。")
                                with gr.Column(scale=1, elem_classes="result-shell result-box"):
                                    batch_raw_results = gr.Textbox(
                                        label="原始结果",
                                        lines=14,
                                        max_lines=14,
                                        interactive=False,
                                        show_copy_button=True,
                                        elem_classes="result-json",
                                    )

                        batch_eval_button.click(
                            fn=evaluate_batch_ui,
                            inputs=[batch_generated_files,
                                    batch_reference_files,
                                    batch_metrics,
                                    batch_device_mode,
                                    batch_compute_fid],
                            outputs=[batch_summary, batch_raw_results, batch_csv_download],
                            api_name=False,
                        )

    return demo


if __name__ == '__main__':
    ui_args = arg_parse()
    ui_args.demo = True
    ui_args.ckpt_dir = 'ckpt/origin'
    ui_args.ttf_path = 'ttf/KaiXinSongA.ttf'

    demo = build_webui(ui_args)

    try:
        demo.queue(default_concurrency_limit=2)
        demo.launch(debug=True, show_api=False, share=False, server_name="0.0.0.0", inbrowser=True)
    except ValueError as launch_error:
        if "localhost is not accessible" in str(launch_error):
            print("Localhost is not accessible. Retrying with share link enabled.")
            demo.launch(debug=True, show_api=False, share=True, inbrowser=True)
        else:
            raise