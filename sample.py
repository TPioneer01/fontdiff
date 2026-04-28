import os
import cv2
import time
import random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed

from src import (FontDiffuserDPMPipeline,
                 FontDiffuserModelDPM,
                 build_ddpm_scheduler,
                 build_unet,
                 build_content_encoder,
                 build_style_encoder)
from utils import (ttf2im,
                   load_ttf,
                   is_char_in_font,
                   save_args_to_yaml,
                   save_single_image,
                   save_image_with_content_style)


def resolve_runtime_device(device_preference="auto"):
    """Resolve runtime device with graceful CPU fallback.

    Supported values: auto, cpu, cuda, cuda:0, gpu.
    """
    preference = str(device_preference).strip().lower() if device_preference is not None else "auto"
    cuda_available = torch.cuda.is_available()

    if preference in ["", "auto"]:
        resolved = "cuda:0" if cuda_available else "cpu"
        message = f"Auto selected '{resolved}' (CUDA available: {cuda_available})."
        return resolved, message

    if preference == "gpu":
        preference = "cuda:0"

    if preference.startswith("cuda"):
        if cuda_available:
            return preference, f"Using requested device '{preference}'."
        return "cpu", "CUDA device requested but CUDA is unavailable. Switched to CPU."

    if preference == "cpu":
        return "cpu", "Using requested device 'cpu'."

    # Final safety net for any unexpected value.
    return ("cuda:0", f"Unknown device '{device_preference}'. Switched to 'cuda:0'.") if cuda_available else (
        "cpu", f"Unknown device '{device_preference}' and CUDA unavailable. Switched to 'cpu'."
    )


def arg_parse():
    from configs.fontdiffuser import get_parser

    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--controlnet", type=bool, default=False, 
                        help="If in demo mode, the controlnet can be added.")
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--content_character", type=str, default=None)
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--save_image_dir", type=str, default=None,
                        help="The saving directory.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Runtime device: auto | cpu | cuda:0")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")
    args = parser.parse_args()
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    args.device, args.device_message = resolve_runtime_device(args.device)
    return args


def image_process(args, content_image=None, style_image=None):
    raw_content_image = None
    raw_style_image = None

    if not args.demo:
        # Read content image and style image
        if args.character_input:
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                return None, None, None, None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
            content_image_pil = content_image.copy()
            raw_content_image = content_image.copy()
        else:
            content_image = Image.open(args.content_image_path).convert('RGB')
            content_image_pil = None
            raw_content_image = content_image.copy()
        style_image = Image.open(args.style_image_path).convert('RGB')
        raw_style_image = style_image.copy()
    else:
        assert style_image is not None, "The style image should not be None."
        if args.character_input:
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                return None, None, None, None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
            raw_content_image = content_image.copy()
        else:
            assert content_image is not None, "The content image should not be None."
            raw_content_image = content_image.copy()
        content_image_pil = None
        raw_style_image = style_image.copy()
        
    ## Dataset transform
    content_inference_transforms = transforms.Compose(
        [transforms.Resize(args.content_image_size, \
                            interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])
    style_inference_transforms = transforms.Compose(
        [transforms.Resize(args.style_image_size, \
                           interpolation=transforms.InterpolationMode.BILINEAR),
         transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])
    content_image = content_inference_transforms(content_image)[None, :]
    style_image = style_inference_transforms(style_image)[None, :]

    # Edge maps are extracted online from existing images for condition injection.
    content_edge_np = cv2.Canny(
        np.array(raw_content_image.convert("L")),
        threshold1=args.edge_canny_low,
        threshold2=args.edge_canny_high,
    )
    style_edge_np = cv2.Canny(
        np.array(raw_style_image.convert("L")),
        threshold1=args.edge_canny_low,
        threshold2=args.edge_canny_high,
    )
    content_edge = torch.from_numpy(content_edge_np).float().unsqueeze(0).unsqueeze(0) / 255.0
    style_edge = torch.from_numpy(style_edge_np).float().unsqueeze(0).unsqueeze(0) / 255.0
    content_edge = torch.nn.functional.interpolate(content_edge, size=args.content_image_size, mode="nearest")
    style_edge = torch.nn.functional.interpolate(style_edge, size=args.style_image_size, mode="nearest")

    return content_image, style_image, content_edge, style_edge, content_image_pil

def load_fontdiffuer_pipeline(args):
    device = torch.device(args.device)

    # Load the model state_dict
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth", map_location=device))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth", map_location=device))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth", map_location=device))
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
        edge_fusion_scale=args.edge_fusion_scale)
    edge_adapter_path = f"{args.ckpt_dir}/edge_adapter.pth"
    if os.path.exists(edge_adapter_path):
        try:
            edge_state = torch.load(edge_adapter_path, map_location=device)
            if "content_adapter" in edge_state and "style_adapter" in edge_state and "fusion_scale" in edge_state:
                model.edge_adapter_content.load_state_dict(edge_state["content_adapter"])
                model.edge_adapter_style.load_state_dict(edge_state["style_adapter"])
                # Safely update edge_fusion_scale using the property setter
                try:
                    model.edge_fusion_scale = edge_state["fusion_scale"].to(device)
                except Exception as e:
                    print(f"[Warning] Could not update edge_fusion_scale: {e}")
            else:
                model.load_state_dict(edge_state, strict=False)
            print("Loaded edge adapter state_dict successfully!")
        except Exception as e:
            print(f"[Warning] Failed to load edge adapter: {e}")
            import traceback
            traceback.print_exc()

    model.to(device)
    print(f"Loaded the model state_dict successfully on {device}!")

    # Load the training ddpm_scheduler.
    train_scheduler = build_ddpm_scheduler(args=args)
    print("Loaded training DDPM scheduler sucessfully!")

    # Load the DPM_Solver to generate the sample.
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("Loaded dpm_solver pipeline sucessfully!")

    return pipe


def sampling(args, pipe, content_image=None, style_image=None):
    if not args.demo:
        os.makedirs(args.save_image_dir, exist_ok=True)
        # saving sampling config
        save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")

    if args.seed:
        set_seed(seed=args.seed)
    
    content_image, style_image, content_edge, style_edge, content_image_pil = image_process(args=args, 
                                                                                              content_image=content_image, 
                                                                                              style_image=style_image)
    if content_image == None:
        print(f"The content_character you provided is not in the ttf. \
                Please change the content_character or you can change the ttf.")
        return None

    with torch.no_grad():
        content_image = content_image.to(args.device)
        style_image = style_image.to(args.device)
        content_edge = content_edge.to(args.device)
        style_edge = style_edge.to(args.device)
        print(f"Sampling by DPM-Solver++ ......")
        start = time.time()
        images = pipe.generate(
            content_images=content_image,
            style_images=style_image,
            content_edges=content_edge,
            style_edges=style_edge,
            batch_size=1,
            order=args.order,
            num_inference_step=args.num_inference_steps,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            t_start=args.t_start,
            t_end=args.t_end,
            dm_size=args.content_image_size,
            algorithm_type=args.algorithm_type,
            skip_type=args.skip_type,
            method=args.method,
            correcting_x0_fn=args.correcting_x0_fn)
        end = time.time()

        if args.save_image:
            print(f"Saving the image ......")
            save_single_image(save_dir=args.save_image_dir, image=images[0])
            if args.character_input:
                save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=content_image_pil,
                                            content_image_path=None,
                                            style_image_path=args.style_image_path,
                                            resolution=args.resolution)
            else:
                save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=None,
                                            content_image_path=args.content_image_path,
                                            style_image_path=args.style_image_path,
                                            resolution=args.resolution)
            print(f"Finish the sampling process, costing time {end - start}s")
        return images[0]


def load_controlnet_pipeline(args,
                             config_path="lllyasviel/sd-controlnet-canny", 
                             ckpt_path="runwayml/stable-diffusion-v1-5"):
    from diffusers import ControlNetModel, AutoencoderKL
    # load controlnet model and pipeline
    from diffusers import StableDiffusionControlNetPipeline, UniPCMultistepScheduler
    dtype = torch.float16 if str(args.device).startswith("cuda") and torch.cuda.is_available() else torch.float32
    controlnet = ControlNetModel.from_pretrained(config_path, 
                                                 torch_dtype=dtype,
                                                 cache_dir=f"{args.ckpt_dir}/controlnet")
    print(f"Loaded ControlNet Model Successfully!")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(ckpt_path, 
                                                             controlnet=controlnet, 
                                                             torch_dtype=dtype,
                                                             cache_dir=f"{args.ckpt_dir}/controlnet_pipeline")
    # faster
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cpu")
    print(f"Loaded ControlNet Pipeline Successfully!")

    return pipe


def controlnet(text_prompt, 
               pil_image,
               pipe):
    image = np.array(pil_image)
    # get canny image
    image = cv2.Canny(image=image, threshold1=100, threshold2=200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    
    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(text_prompt, 
                 num_inference_steps=50, 
                 generator=generator, 
                 image=canny_image,
                 output_type='pil').images[0]
    return image


def load_instructpix2pix_pipeline(args,
                                  ckpt_path="timbrooks/instruct-pix2pix"):
    from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
    dtype = torch.float16 if str(args.device).startswith("cuda") and torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(ckpt_path, 
                                                                  torch_dtype=dtype)
    pipe.to(args.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe

def instructpix2pix(pil_image, text_prompt, pipe):
    image = pil_image.resize((512, 512))
    seed = random.randint(0, 10000)
    generator = torch.manual_seed(seed)
    image = pipe(prompt=text_prompt, image=image, generator=generator, 
                 num_inference_steps=20, image_guidance_scale=1.1).images[0]

    return image


def inference_on_dataset_samples(args, pipe, train_dataset, device, num_samples=3, seed=42):
    """
    Perform inference on random samples from training dataset during training.
    
    Args:
        args: Training arguments
        pipe: FontDiffuserDPMPipeline instance
        train_dataset: FontDataset instance for sampling
        device: Device to run inference on
        num_samples: Number of samples to infer
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (generated_images, content_images_pil, style_images_pil, sample_indices)
    """
    import random as py_random
    
    # Set seed for reproducibility
    py_random.seed(seed)
    dataset_size = len(train_dataset)
    sample_indices = py_random.sample(range(dataset_size), min(num_samples, dataset_size))
    
    generated_images = []
    content_images_pil = []
    style_images_pil = []
    
    with torch.no_grad():
        for sample_idx in sample_indices:
            # Load sample from dataset
            sample = train_dataset[sample_idx]
            
            # Extract and prepare tensors
            content_image = sample["content_image"].unsqueeze(0).to(device)  # Add batch dim
            style_image = sample["style_image"].unsqueeze(0).to(device)
            content_edge = sample["content_edge"].unsqueeze(0).to(device)
            style_edge = sample["style_edge"].unsqueeze(0).to(device)
            
            # Convert normalized tensors back to PIL for display
            # (training tensors are normalized to [-1, 1])
            def tensor_to_pil(tensor_normalized):
                # tensor_normalized is in [-1, 1] range after training transforms
                tensor_img = ((tensor_normalized + 1) / 2).clamp(0, 1)  # Convert to [0, 1]
                tensor_img = (tensor_img * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                return Image.fromarray(tensor_img)
            
            content_pil = tensor_to_pil(content_image.squeeze(0))
            style_pil = tensor_to_pil(style_image.squeeze(0))
            
            content_images_pil.append(content_pil)
            style_images_pil.append(style_pil)
            
            try:
                # Run inference
                print(f"[Inference] Sampling for dataset index {sample_idx}...")
                gen_images = pipe.generate(
                    content_images=content_image,
                    style_images=style_image,
                    content_edges=content_edge,
                    style_edges=style_edge,
                    batch_size=1,
                    order=args.order,
                    num_inference_step=args.num_inference_steps,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                    t_start=args.t_start,
                    t_end=args.t_end,
                    dm_size=args.content_image_size,
                    algorithm_type=args.algorithm_type,
                    skip_type=args.skip_type,
                    method=args.method,
                    correcting_x0_fn=args.correcting_x0_fn
                )
                generated_images.append(gen_images[0])
            except Exception as e:
                print(f"[Inference Error] Failed for sample {sample_idx}: {e}")
                # Add blank image as placeholder on error
                generated_images.append(Image.new('RGB', (args.resolution, args.resolution)))
    
    return generated_images, content_images_pil, style_images_pil, sample_indices


if __name__=="__main__":
    args = arg_parse()
    
    # load fontdiffuser pipeline
    pipe = load_fontdiffuer_pipeline(args=args)
    out_image = sampling(args=args, pipe=pipe)
