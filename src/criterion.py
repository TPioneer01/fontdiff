import torch
import torch.nn as nn
import torchvision 
import torch.nn.functional as F

try:
    import kornia
except ImportError:
    kornia = None


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        for i in range(3):
            for param in getattr(self, f'enc_{i+1:d}').parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, f'enc_{i+1:d}')
            results.append(func(results[-1]))
        return results[1:]


class ContentPerceptualLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.VGG = VGG16()

    def calculate_loss(self, generated_images, target_images, device):
        self.VGG = self.VGG.to(device)

        generated_features = self.VGG(generated_images)
        target_features = self.VGG(target_images)

        perceptual_loss = 0
        perceptual_loss += torch.mean((target_features[0] - generated_features[0]) ** 2)
        perceptual_loss += torch.mean((target_features[1] - generated_features[1]) ** 2)
        perceptual_loss += torch.mean((target_features[2] - generated_features[2]) ** 2)
        perceptual_loss /= 3
        return perceptual_loss


class EdgeConsistencyLoss(nn.Module):
    """Encourage generated glyph structure to match target edge layout."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    @staticmethod
    def _to_gray(image):
        if image.shape[1] == 1:
            return image
        return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]

    def _sobel_fallback(self, gray_image):
        device = gray_image.device
        dtype = gray_image.dtype
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=device,
            dtype=dtype,
        ).view(1, 1, 3, 3)
        grad_x = F.conv2d(gray_image, kernel_x, padding=1)
        grad_y = F.conv2d(gray_image, kernel_y, padding=1)
        return torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)

    def _edge_map(self, image):
        gray = self._to_gray(image)
        if kornia is not None:
            gradients = kornia.filters.spatial_gradient(gray)
            grad_x = gradients[:, :, 0]
            grad_y = gradients[:, :, 1]
            return torch.sqrt(grad_x * grad_x + grad_y * grad_y + self.eps)
        return self._sobel_fallback(gray)

    def forward(self, generated_images, target_images, target_edge_maps=None):
        pred_edges = self._edge_map(generated_images)
        if target_edge_maps is not None:
            gt_edges = target_edge_maps
        else:
            gt_edges = self._edge_map(target_images)
        return F.l1_loss(pred_edges, gt_edges)
