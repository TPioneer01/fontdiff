"""
评价指标模块：支持L1、L2、RMSE、PSNR、SSIM、LPIPS、FID

支持单样本评价和多样本批量评价
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Tuple
import time
import warnings

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    warnings.warn("scikit-image not installed. SSIM metrics will not be available.")
    ssim = None

try:
    import lpips as lpips_lib
except ImportError:
    warnings.warn("lpips not installed. LPIPS metrics will not be available.")
    lpips_lib = None

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths
except ImportError:
    warnings.warn("pytorch_fid not installed. FID metrics will not be available.")
    calculate_fid_given_paths = None


class SingleSampleEvaluator:
    """单样本评价类
    
    支持指标：L1、L2、RMSE、PSNR、SSIM、LPIPS
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # 初始化LPIPS模型
        self.lpips_model = None
        if lpips_lib is not None:
            try:
                self.lpips_model = lpips_lib.LPIPS(net='alex', version='0.1').to(device)
                self.lpips_model.eval()
            except Exception as e:
                warnings.warn(f"Failed to load LPIPS model: {e}")
    
    def _prepare_tensors(
        self, 
        img1: Union[torch.Tensor, np.ndarray], 
        img2: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """准备张量，确保形状和范围一致
        
        输入可以是：
        - (H, W) 灰度图
        - (3, H, W) 或 (H, W, 3) RGB图
        - 值范围 [0, 255] 或 [0, 1]
        
        输出：(B=1, C, H, W) 范围 [0, 1]
        """
        # 转换为张量
        if Image is not None and isinstance(img1, Image.Image):
            img1 = np.array(img1)
        if isinstance(img1, np.ndarray):
            img1 = torch.from_numpy(img1).float()
        else:
            img1 = img1.float()
            
        if Image is not None and isinstance(img2, Image.Image):
            img2 = np.array(img2)
        if isinstance(img2, np.ndarray):
            img2 = torch.from_numpy(img2).float()
        else:
            img2 = img2.float()
        
        # 处理形状
        if img1.dim() == 2:  # (H, W) -> (1, 1, H, W)
            img1 = img1.unsqueeze(0).unsqueeze(0)
        elif img1.dim() == 3:
            if img1.shape[0] in [1, 3]:  # (C, H, W) -> (1, C, H, W)
                img1 = img1.unsqueeze(0)
            else:  # (H, W, C) -> (1, C, H, W)
                img1 = img1.permute(2, 0, 1).unsqueeze(0)
        
        if img2.dim() == 2:  # (H, W) -> (1, 1, H, W)
            img2 = img2.unsqueeze(0).unsqueeze(0)
        elif img2.dim() == 3:
            if img2.shape[0] in [1, 3]:  # (C, H, W) -> (1, C, H, W)
                img2 = img2.unsqueeze(0)
            else:  # (H, W, C) -> (1, C, H, W)
                img2 = img2.permute(2, 0, 1).unsqueeze(0)
        
        # 尺寸不一致时，自动把参考图缩放到生成图尺寸，保证评价可以继续进行
        if img1.shape[-2:] != img2.shape[-2:]:
            target_height, target_width = img1.shape[-2:]
            interpolation_mode = "bilinear" if img2.shape[1] != 1 else "nearest"
            img2 = F.interpolate(
                img2,
                size=(target_height, target_width),
                mode=interpolation_mode,
                align_corners=False if interpolation_mode != "nearest" else None,
            )
        
        # 归一化到 [0, 1] 范围
        if img1.max() > 1.0:
            img1 = img1 / 255.0
        if img2.max() > 1.0:
            img2 = img2 / 255.0
        
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        return img1, img2
    
    def calculate_l1(
        self, 
        img1: Union[torch.Tensor, np.ndarray], 
        img2: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """计算L1距离 (MAE)"""
        img1, img2 = self._prepare_tensors(img1, img2)
        l1 = torch.mean(torch.abs(img1 - img2)).item()
        return l1
    
    def calculate_l2(
        self, 
        img1: Union[torch.Tensor, np.ndarray], 
        img2: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """计算L2距离 (MSE)"""
        img1, img2 = self._prepare_tensors(img1, img2)
        l2 = torch.mean((img1 - img2) ** 2).item()
        return l2
    
    def calculate_rmse(
        self, 
        img1: Union[torch.Tensor, np.ndarray], 
        img2: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """计算RMSE (均方根误差)"""
        l2 = self.calculate_l2(img1, img2)
        rmse = np.sqrt(l2)
        return rmse
    
    def calculate_psnr(
        self, 
        img1: Union[torch.Tensor, np.ndarray], 
        img2: Union[torch.Tensor, np.ndarray],
        max_pixel: float = 1.0
    ) -> float:
        """计算PSNR (峰值信噪比)
        
        Args:
            max_pixel: 最大像素值，通常为1.0（归一化）或255.0（8-bit）
        """
        img1, img2 = self._prepare_tensors(img1, img2)
        
        mse = torch.mean((img1 - img2) ** 2).item()
        if mse == 0:
            return float('inf')
        
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def calculate_ssim(
        self, 
        img1: Union[torch.Tensor, np.ndarray], 
        img2: Union[torch.Tensor, np.ndarray],
        data_range: float = 1.0
    ) -> float:
        """计算SSIM (结构相似性指数)
        
        Args:
            data_range: 数据范围，通常为1.0（归一化）或255.0（8-bit）
        """
        if ssim is None:
            raise ImportError("scikit-image is required for SSIM calculation. Install with: pip install scikit-image")
        
        img1, img2 = self._prepare_tensors(img1, img2)
        
        # 转换为numpy并移除batch维度
        img1_np = img1.squeeze(0).cpu().numpy()
        img2_np = img2.squeeze(0).cpu().numpy()
        
        # 如果是单通道，转为2D
        channel_axis = 0
        if img1_np.shape[0] == 1:
            img1_np = img1_np[0]
            img2_np = img2_np[0]
            channel_axis = None
        
        ssim_value = ssim(img1_np, img2_np, data_range=data_range, channel_axis=channel_axis)
        return float(ssim_value)
    
    def calculate_lpips(
        self, 
        img1: Union[torch.Tensor, np.ndarray], 
        img2: Union[torch.Tensor, np.ndarray]
    ) -> float:
        """计算LPIPS (学习感知图像块相似性)"""
        if self.lpips_model is None:
            raise ImportError("lpips is required for LPIPS calculation. Install with: pip install lpips")
        
        img1, img2 = self._prepare_tensors(img1, img2)
        
        # LPIPS期望输入范围 [-1, 1]
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1
        
        with torch.no_grad():
            lpips_value = self.lpips_model(img1, img2).item()
        
        return lpips_value
    
    def evaluate(
        self, 
        img1: Union[torch.Tensor, np.ndarray], 
        img2: Union[torch.Tensor, np.ndarray],
        metrics: List[str] = None
    ) -> Dict[str, float]:
        """评价单样本
        
        Args:
            img1: 生成的图像
            img2: 参考图像
            metrics: 要计算的指标列表，默认为['L1', 'L2', 'RMSE', 'PSNR', 'SSIM', 'LPIPS']
        
        Returns:
            指标字典
        """
        if metrics is None:
            metrics = ['L1', 'L2', 'RMSE', 'PSNR', 'SSIM', 'LPIPS']
        
        results = {}
        metrics = [m.upper() for m in metrics]
        
        try:
            if 'L1' in metrics:
                results['L1'] = self.calculate_l1(img1, img2)
            
            if 'L2' in metrics:
                results['L2'] = self.calculate_l2(img1, img2)
            
            if 'RMSE' in metrics:
                results['RMSE'] = self.calculate_rmse(img1, img2)
            
            if 'PSNR' in metrics:
                results['PSNR'] = self.calculate_psnr(img1, img2)
            
            if 'SSIM' in metrics:
                results['SSIM'] = self.calculate_ssim(img1, img2)
            
            if 'LPIPS' in metrics:
                results['LPIPS'] = self.calculate_lpips(img1, img2)
        
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise
        
        return results


class BatchEvaluator:
    """多样本批量评价类
    
    支持指标：L1、L2、RMSE、PSNR、SSIM、LPIPS、FID
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.single_evaluator = SingleSampleEvaluator(device=device)
    
    def evaluate_batch(
        self, 
        img_list1: List[Union[torch.Tensor, np.ndarray]],
        img_list2: List[Union[torch.Tensor, np.ndarray]],
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """批量评价图像（不包括FID）
        
        Args:
            img_list1: 生成图像列表
            img_list2: 参考图像列表
            metrics: 要计算的指标列表，默认为['L1', 'L2', 'RMSE', 'PSNR', 'SSIM', 'LPIPS']
        
        Returns:
            包含每个指标的均值、标准差和每样本值的字典
        """
        if metrics is None:
            metrics = ['L1', 'L2', 'RMSE', 'PSNR', 'SSIM', 'LPIPS']
        
        assert len(img_list1) == len(img_list2), \
            f"Image lists have different lengths: {len(img_list1)} vs {len(img_list2)}"
        
        metrics = [m.upper() for m in metrics]
        n_samples = len(img_list1)
        
        # 初始化结果字典
        all_results = {metric: [] for metric in metrics}
        
        print(f"正在评价 {n_samples} 个样本...")
        start_time = time.perf_counter()
        
        # 逐样本评价
        for idx, (img1, img2) in enumerate(zip(img_list1, img_list2)):
            sample_results = self.single_evaluator.evaluate(img1, img2, metrics)
            
            for metric, value in sample_results.items():
                all_results[metric].append(value)
            
            processed = idx + 1
            if processed % max(1, n_samples // 10) == 0 or processed == n_samples:
                elapsed = time.perf_counter() - start_time
                speed = processed / elapsed if elapsed > 0 else 0.0
                remaining = max(0, n_samples - processed)
                eta = (remaining / speed) if speed > 0 else None
                eta_text = "unknown" if eta is None else f"{eta:.1f}s"
                print(f"已完成 {processed}/{n_samples} 样本 | 速率 {speed:.2f} 样本/秒 | ETA {eta_text}")
        
        # 计算统计量
        final_results = {}
        for metric in metrics:
            values = np.array(all_results[metric])
            final_results[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
        
        return final_results
    
    def calculate_fid(
        self,
        path_fake: Union[str, Path],
        path_real: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4
    ) -> float:
        """计算FID (Fréchet Inception Distance)
        
        只支持多样本评价
        
        Args:
            path_fake: 生成图像所在目录
            path_real: 真实图像所在目录
            batch_size: 批大小
            num_workers: 数据加载器工作线程数
        
        Returns:
            FID分数
        """
        if calculate_fid_given_paths is None:
            raise ImportError("pytorch_fid is required for FID calculation. Install with: pip install pytorch_fid")
        
        path_fake = str(Path(path_fake).resolve())
        path_real = str(Path(path_real).resolve())
        
        print(f"正在计算FID...")
        print(f"  生成图像路径: {path_fake}")
        print(f"  真实图像路径: {path_real}")
        
        try:
            fid_value = calculate_fid_given_paths(
                [path_fake, path_real],
                batch_size=batch_size,
                device=self.device,
                dims=2048,
                num_workers=num_workers
            )
            return float(fid_value)
        except Exception as e:
            print(f"FID计算失败: {e}")
            raise
    
    def evaluate_with_fid(
        self,
        img_list1: List[Union[torch.Tensor, np.ndarray]],
        img_list2: List[Union[torch.Tensor, np.ndarray]],
        path_fake: Union[str, Path],
        path_real: Union[str, Path],
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """完整评价（包括FID）
        
        Args:
            img_list1: 生成图像列表
            img_list2: 参考图像列表
            path_fake: 生成图像所在目录（用于FID）
            path_real: 真实图像所在目录（用于FID）
            metrics: 要计算的指标列表，默认为['L1', 'L2', 'RMSE', 'PSNR', 'SSIM', 'LPIPS', 'FID']
        
        Returns:
            包含所有指标的字典
        """
        if metrics is None:
            metrics = ['L1', 'L2', 'RMSE', 'PSNR', 'SSIM', 'LPIPS', 'FID']
        
        metrics = [m.upper() for m in metrics]
        
        # 计算非FID指标
        other_metrics = [m for m in metrics if m != 'FID']
        results = self.evaluate_batch(img_list1, img_list2, other_metrics)
        
        # 计算FID
        if 'FID' in metrics:
            fid_value = self.calculate_fid(path_fake, path_real)
            results['FID'] = {
                'mean': fid_value,
                'std': 0.0,
                'min': fid_value,
                'max': fid_value,
            }
        
        return results


# 便利函数
def evaluate_single_sample(
    img1: Union[torch.Tensor, np.ndarray],
    img2: Union[torch.Tensor, np.ndarray],
    metrics: List[str] = None,
    device: str = None
) -> Dict[str, float]:
    """便利函数：评价单样本
    
    Args:
        img1: 生成的图像
        img2: 参考图像
        metrics: 指标列表
        device: 设备
    
    Returns:
        指标字典
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    evaluator = SingleSampleEvaluator(device=device)
    return evaluator.evaluate(img1, img2, metrics)


def evaluate_batch_samples(
    img_list1: List[Union[torch.Tensor, np.ndarray]],
    img_list2: List[Union[torch.Tensor, np.ndarray]],
    metrics: List[str] = None,
    device: str = None
) -> Dict[str, Dict[str, float]]:
    """便利函数：评价多样本
    
    Args:
        img_list1: 生成图像列表
        img_list2: 参考图像列表
        metrics: 指标列表
        device: 设备
    
    Returns:
        统计量字典
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    evaluator = BatchEvaluator(device=device)
    return evaluator.evaluate_batch(img_list1, img_list2, metrics)


def calculate_fid_score(
    path_fake: Union[str, Path],
    path_real: Union[str, Path],
    device: str = None
) -> float:
    """便利函数：计算FID
    
    Args:
        path_fake: 生成图像目录
        path_real: 真实图像目录
        device: 设备
    
    Returns:
        FID分数
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    evaluator = BatchEvaluator(device=device)
    return evaluator.calculate_fid(path_fake, path_real)
