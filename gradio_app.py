import os
import random
from datetime import datetime

import gradio as gr

from sample import (
    arg_parse,
    sampling,
    load_fontdiffuer_pipeline,
    resolve_runtime_device,
)


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


def run_fontdiffuer(source_image,
                    source_characters,
                    reference_image,
                    sampling_step,
                    guidance_scale,
                    device_mode):
    global pipe

    resolved_device, device_message = resolve_runtime_device(device_mode)
    if resolved_device != args.device:
        args.device = resolved_device
        pipe = load_fontdiffuer_pipeline(args=args)

    args.num_inference_steps = int(sampling_step)
    args.guidance_scale = guidance_scale
    args.save_image = True

    run_root = os.path.join("outputs", "gradio_runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_root, exist_ok=True)

    if source_image is not None:
        args.character_input = False
        args.content_character = None
        args.save_image_dir = os.path.join(run_root, "source_image")
        os.makedirs(args.save_image_dir, exist_ok=True)
        args.seed = random.randint(0, 10000)
        generated_image = sampling(
            args=args,
            pipe=pipe,
            content_image=source_image,
            style_image=reference_image,
        )
        status_lines = [
            f"{device_message} Current runtime device: {args.device}.",
            f"Saved 1 result to {args.save_image_dir}.",
        ]
        return [(generated_image, "source_image")], "\n".join(status_lines), args.save_image_dir

    source_characters_list = _parse_source_characters(source_characters)
    if not source_characters_list:
        return [], f"{device_message} Current runtime device: {args.device}.\nPlease provide at least one source character.", run_root

    results = []
    status_lines = [f"{device_message} Current runtime device: {args.device}."]

    for index, character in enumerate(source_characters_list, start=1):
        args.character_input = True
        args.content_character = character
        args.save_image_dir = os.path.join(run_root, f"{index:02d}_{_safe_dirname(character)}")
        os.makedirs(args.save_image_dir, exist_ok=True)
        args.seed = random.randint(0, 10000)

        generated_image = sampling(
            args=args,
            pipe=pipe,
            content_image=None,
            style_image=reference_image,
        )
        if generated_image is not None:
            results.append((generated_image, character))
            status_lines.append(f"{index:02d}. {character} -> {args.save_image_dir}")
        else:
            status_lines.append(f"{index:02d}. {character} -> skipped (character not in font).")

    status_lines.append(f"Saved {len(results)} result(s) under {run_root}.")
    return results, "\n".join(status_lines), run_root


if __name__ == '__main__':
    args = arg_parse()
    args.demo = True
    args.ckpt_dir = 'ckpt'
    args.ttf_path = 'ttf/KaiXinSongA.ttf'

    # load fontdiffuer pipeline
    pipe = load_fontdiffuer_pipeline(args=args)
    device_status_default = f"{args.device_message} Current runtime device: {args.device}."

    with gr.Blocks() as demo:
        gr.Markdown("# FontDiffuser")
        gr.Markdown("输入一个参考图或一组 Source Character，点击运行后会逐个推理并把结果保存到本地目录。")

        with gr.Row():
            source_image = gr.Image(width=320, label='Source Image (optional)', image_mode='RGB', type='pil')
            reference_image = gr.Image(width=320, label='Reference Image', image_mode='RGB', type='pil')

        source_characters = gr.Textbox(
            value='中\n国\n矿\n业\n大\n学\n',
            lines=6,
            label='Source Characters',
            placeholder='每行一个字符，也支持用英文逗号分隔，例如: 中,国,矿,业,大,学',
        )

        with gr.Row():
            device_mode = gr.Dropdown(
                choices=["auto", "cpu", "cuda:0"],
                value=args.device,
                label="Runtime Device",
            )
            sampling_step = gr.Slider(20, 50, value=20, step=10, label="Sampling Step")
            guidance_scale = gr.Slider(1, 12, value=7.5, step=0.5, label="Guidance Scale")

        run_button = gr.Button('Run FontDiffuser')

        output_gallery = gr.Gallery(label='Generated Results', columns=2, height=360)
        runtime_status = gr.Textbox(value=device_status_default, label='Runtime Status', interactive=False, lines=4)
        save_dir_text = gr.Textbox(value='', label='Saved Directory', interactive=False)

        run_button.click(
            fn=run_fontdiffuer,
            inputs=[source_image,
                    source_characters,
                    reference_image,
                    sampling_step,
                    guidance_scale,
                    device_mode],
            outputs=[output_gallery, runtime_status, save_dir_text],
            api_name=False)

    try:
        demo.launch(debug=True, show_api=False, share=False, server_name="127.0.0.1", inbrowser=True)
    except ValueError as launch_error:
        if "localhost is not accessible" in str(launch_error):
            print("Localhost is not accessible. Retrying with share link enabled.")
            demo.launch(debug=True, show_api=False, share=True, inbrowser=True)
        else:
            raise