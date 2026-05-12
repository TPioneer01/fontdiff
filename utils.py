import os
import cv2
import yaml
import copy
import pygame
import numpy as np
from PIL import Image
from fontTools.ttLib import TTFont

import torch
import torchvision.transforms as transforms

def save_args_to_yaml(args, output_file):
    # Convert args namespace to a dictionary
    args_dict = vars(args)

    # Write the dictionary to a YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)


def save_single_image(save_dir, image):

    save_path = f"{save_dir}/out_single.png"
    image.save(save_path)


def save_image_with_content_style(save_dir, image, content_image_pil, content_image_path, style_image_pil, style_image_path, resolution):
    def _as_rgb_resized(img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (resolution, resolution):
            img = img.resize((resolution, resolution), Image.BILINEAR)
        return img

    new_image = Image.new('RGB', (resolution*3, resolution))
    if content_image_pil is not None:
        content_image = _as_rgb_resized(content_image_pil)
    else:
        content_image = _as_rgb_resized(Image.open(content_image_path))
    if style_image_pil is not None:
        style_image = _as_rgb_resized(style_image_pil)
    else:
        style_image = _as_rgb_resized(Image.open(style_image_path))

    image = _as_rgb_resized(image)

    new_image.paste(content_image, (0, 0))
    new_image.paste(style_image, (resolution, 0))
    new_image.paste(image, (resolution*2, 0))

    save_path = f"{save_dir}/out_with_cs.jpg"
    new_image.save(save_path)


def save_inference_batch_results(save_dir, generated_images, content_images_pil, style_images_pil, 
                                  sample_indices=None, resolution=96):
    """
    Save batch of inference results with content, style, and generated images side-by-side.
    
    Args:
        save_dir: Directory to save results
        generated_images: Tensor or list of PIL Images (generated outputs)
        content_images_pil: List of PIL Images (content inputs)
        style_images_pil: List of PIL Images (style inputs)
        sample_indices: List of indices for naming (e.g., [0, 5, 10])
        resolution: Image resolution
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Handle tensor to PIL conversion if needed
    if isinstance(generated_images, torch.Tensor):
        generated_images = [
            Image.fromarray((img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            for img in generated_images
        ]
    
    num_samples = len(generated_images)
    if sample_indices is None:
        sample_indices = list(range(num_samples))
    
    for idx, (gen_img, cnt_img, sty_img, sample_id) in enumerate(
        zip(generated_images, content_images_pil, style_images_pil, sample_indices)
    ):
        # Create concatenated image: content | style | generated
        concat_image = Image.new('RGB', (resolution * 3, resolution))
        
        # Ensure PIL format
        if not isinstance(cnt_img, Image.Image):
            cnt_img = Image.fromarray(cnt_img) if isinstance(cnt_img, np.ndarray) else cnt_img
        if not isinstance(sty_img, Image.Image):
            sty_img = Image.fromarray(sty_img) if isinstance(sty_img, np.ndarray) else sty_img
        if not isinstance(gen_img, Image.Image):
            gen_img = Image.fromarray(gen_img) if isinstance(gen_img, np.ndarray) else gen_img
        
        # Resize to match resolution if needed
        cnt_img = cnt_img.resize((resolution, resolution), Image.BILINEAR) if cnt_img.size != (resolution, resolution) else cnt_img
        sty_img = sty_img.resize((resolution, resolution), Image.BILINEAR) if sty_img.size != (resolution, resolution) else sty_img
        gen_img = gen_img.resize((resolution, resolution), Image.BILINEAR) if gen_img.size != (resolution, resolution) else gen_img
        
        concat_image.paste(cnt_img, (0, 0))
        concat_image.paste(sty_img, (resolution, 0))
        concat_image.paste(gen_img, (resolution * 2, 0))
        
        # Save with sample index
        concat_image.save(f"{save_dir}/inference_sample_{sample_id:03d}.jpg")


def x0_from_epsilon(scheduler, noise_pred, x_t, timesteps):
    """Return the x_0 from epsilon
    """
    batch_size = noise_pred.shape[0]
    for i in range(batch_size):
        noise_pred_i = noise_pred[i]
        noise_pred_i = noise_pred_i[None, :]
        t = timesteps[i]
        x_t_i = x_t[i]
        x_t_i = x_t_i[None, :]

        pred_original_sample_i = scheduler.step(
            model_output=noise_pred_i,
            timestep=t,
            sample=x_t_i,
            # predict_epsilon=True,
            generator=None,
            return_dict=True,
        ).pred_original_sample
        if i == 0:
            pred_original_sample = pred_original_sample_i
        else:
            pred_original_sample = torch.cat((pred_original_sample, pred_original_sample_i), dim=0)

    return pred_original_sample


def reNormalize_img(pred_original_sample):
    pred_original_sample = (pred_original_sample / 2 + 0.5).clamp(0, 1)
    
    return pred_original_sample


def normalize_mean_std(image):
    transforms_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = transforms_norm(image)

    return image


def is_char_in_font(font_path, char):
    TTFont_font = TTFont(font_path)
    cmap = TTFont_font['cmap']
    for subtable in cmap.tables:
        if ord(char) in subtable.cmap:
            return True
    return False


def load_ttf(ttf_path, fsize=128):
    pygame.init()

    font = pygame.freetype.Font(ttf_path, size=fsize)
    return font


def ttf2im(font, char, fsize=128):
    
    try:
        surface, _ = font.render(char)
    except:
        print("No glyph for char {}".format(char))
        return
    bg = np.full((fsize, fsize), 255)
    imo = pygame.surfarray.pixels_alpha(surface).transpose(1, 0)
    imo = 255 - np.array(Image.fromarray(imo))
    im = copy.deepcopy(bg)
    h, w = imo.shape[:2]
    if h > fsize:
        h, w = fsize, round(w*fsize/h)
        imo = cv2.resize(imo, (w, h))
    if w > fsize:
        h, w = round(h*fsize/w), fsize
        imo = cv2.resize(imo, (w, h))
    x, y = round((fsize-w)/2), round((fsize-h)/2)
    im[y:h+y, x:x+w] = imo
    pil_im = Image.fromarray(im.astype('uint8')).convert('RGB')
    
    return pil_im
