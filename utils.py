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
import torch.nn.functional as F

def save_args_to_yaml(args, output_file):
    # Convert args namespace to a dictionary
    args_dict = vars(args)

    # Write the dictionary to a YAML file
    with open(output_file, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)


def save_single_image(save_dir, image):

    save_path = f"{save_dir}/out_single.png"
    image.save(save_path)


def save_image_with_content_style(save_dir, image, content_image_pil, content_image_path, style_image_path, resolution):
    
    new_image = Image.new('RGB', (resolution*3, resolution))
    if content_image_pil is not None:
        content_image = content_image_pil
    else:
        content_image = Image.open(content_image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    style_image = Image.open(style_image_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)

    new_image.paste(content_image, (0, 0))
    new_image.paste(style_image, (resolution, 0))
    new_image.paste(image, (resolution*2, 0))

    save_path = f"{save_dir}/out_with_cs.jpg"
    new_image.save(save_path)


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


def canny_edge_from_pil(image, resolution, low_threshold=100, high_threshold=200):
    """Extract Canny edge map from PIL image and return tensor in [0, 1] with shape [1, H, W]."""
    gray = np.array(image.convert("L").resize((resolution, resolution), Image.BILINEAR))
    edges = cv2.Canny(gray, threshold1=low_threshold, threshold2=high_threshold)
    edges = torch.from_numpy(edges).float().unsqueeze(0) / 255.0
    return edges


def edge_from_tensor(image_tensor):
    """Differentiable edge magnitude map from image tensor.

    Args:
        image_tensor: tensor in shape [B, 3, H, W], expected in [-1, 1] or [0, 1].
    Returns:
        edge magnitude map in shape [B, 1, H, W], in [0, 1].
    """
    if image_tensor.min() < 0:
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)

    gray = 0.2989 * image_tensor[:, 0:1] + 0.5870 * image_tensor[:, 1:2] + 0.1140 * image_tensor[:, 2:3]
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
    magnitude = magnitude / (magnitude.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return magnitude.clamp(0, 1)


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
