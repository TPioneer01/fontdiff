"""
评价模块使用示例

演示：
1. 单样本评价
2. 多样本批量评价
3. FID评价
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from evaluation import (
    SingleSampleEvaluator,
    BatchEvaluator,
    evaluate_single_sample,
    evaluate_batch_samples,
    calculate_fid_score
)


# ============================================================================
# 示例 1: 单样本评价
# ============================================================================

def example_single_sample():
    """单样本评价示例"""
    print("\n" + "="*80)
    print("示例 1: 单样本评价")
    print("="*80)
    
    # 创建评价器
    evaluator = SingleSampleEvaluator(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟图像（实际应用中应从文件读取）
    # 方法1：使用numpy数组
    img1 = np.random.rand(256, 256, 3)  # (H, W, C)，值范围 [0, 1]
    img2 = np.random.rand(256, 256, 3)
    
    # 方法2：使用PyTorch张量
    # img1 = torch.rand(3, 256, 256)  # (C, H, W)
    # img2 = torch.rand(3, 256, 256)
    
    # 评价单个指标
    print("\n单个指标计算：")
    l1 = evaluator.calculate_l1(img1, img2)
    print(f"  L1:   {l1:.6f}")
    
    l2 = evaluator.calculate_l2(img1, img2)
    print(f"  L2:   {l2:.6f}")
    
    rmse = evaluator.calculate_rmse(img1, img2)
    print(f"  RMSE: {rmse:.6f}")
    
    psnr = evaluator.calculate_psnr(img1, img2)
    print(f"  PSNR: {psnr:.2f} dB")
    
    ssim = evaluator.calculate_ssim(img1, img2)
    print(f"  SSIM: {ssim:.6f}")
    
    lpips = evaluator.calculate_lpips(img1, img2)
    print(f"  LPIPS: {lpips:.6f}")
    
    # 一次性计算所有指标
    print("\n一次性计算所有指标：")
    results = evaluator.evaluate(img1, img2)
    for metric, value in results.items():
        print(f"  {metric}: {value:.6f}")
    
    # 计算指定指标
    print("\n计算指定指标（L1, PSNR, SSIM）：")
    results = evaluator.evaluate(img1, img2, metrics=['L1', 'PSNR', 'SSIM'])
    for metric, value in results.items():
        print(f"  {metric}: {value:.6f}")


# ============================================================================
# 示例 2: 多样本批量评价
# ============================================================================

def example_batch_evaluation():
    """多样本批量评价示例"""
    print("\n" + "="*80)
    print("示例 2: 多样本批量评价")
    print("="*80)
    
    # 创建评价器
    evaluator = BatchEvaluator(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟多个图像对
    n_samples = 10
    img_list1 = [np.random.rand(256, 256, 3) for _ in range(n_samples)]
    img_list2 = [np.random.rand(256, 256, 3) for _ in range(n_samples)]
    
    # 批量评价
    results = evaluator.evaluate_batch(img_list1, img_list2)
    
    print("\n批量评价结果统计：")
    print(f"样本数: {n_samples}\n")
    for metric, stats in results.items():
        print(f"{metric}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std:  {stats['std']:.6f}")
        print(f"  Min:  {stats['min']:.6f}")
        print(f"  Max:  {stats['max']:.6f}")
        print()


# ============================================================================
# 示例 3: 使用便利函数
# ============================================================================

def example_convenience_functions():
    """便利函数使用示例"""
    print("\n" + "="*80)
    print("示例 3: 便利函数")
    print("="*80)
    
    # 单样本评价
    img1 = np.random.rand(256, 256, 3)
    img2 = np.random.rand(256, 256, 3)
    
    results = evaluate_single_sample(img1, img2)
    print("\n使用便利函数计算单样本：")
    for metric, value in results.items():
        print(f"  {metric}: {value:.6f}")
    
    # 多样本评价
    n_samples = 5
    img_list1 = [np.random.rand(256, 256, 3) for _ in range(n_samples)]
    img_list2 = [np.random.rand(256, 256, 3) for _ in range(n_samples)]
    
    results = evaluate_batch_samples(img_list1, img_list2)
    print("\n使用便利函数计算多样本：")
    for metric, stats in results.items():
        print(f"{metric} mean: {stats['mean']:.6f}")


# ============================================================================
# 示例 4: 从文件加载图像并评价
# ============================================================================

def example_with_image_files():
    """从文件加载图像的示例"""
    print("\n" + "="*80)
    print("示例 4: 从文件加载图像并评价")
    print("="*80)
    
    evaluator = SingleSampleEvaluator()
    
    # 示例路径（需要自己替换为实际路径）
    # img_path1 = "path/to/generated/image.png"
    # img_path2 = "path/to/reference/image.png"
    
    # # 加载图像
    # img1 = Image.open(img_path1).convert('RGB')
    # img2 = Image.open(img_path2).convert('RGB')
    
    # # 转换为numpy数组
    # img1 = np.array(img1)
    # img2 = np.array(img2)
    
    # # 如果图像值范围是 [0, 255]，评价时会自动归一化
    # results = evaluator.evaluate(img1, img2)
    
    print("\n示例代码（未执行）:")
    print("""
    from PIL import Image
    import numpy as np
    
    # 从文件加载
    img1 = Image.open('generated.png').convert('RGB')
    img2 = Image.open('reference.png').convert('RGB')
    
    img1 = np.array(img1)  # 值范围 [0, 255]
    img2 = np.array(img2)
    
    # 评价
    evaluator = SingleSampleEvaluator()
    results = evaluator.evaluate(img1, img2)
    
    print(f"L1: {results['L1']:.6f}")
    print(f"PSNR: {results['PSNR']:.2f} dB")
    print(f"SSIM: {results['SSIM']:.6f}")
    """)


# ============================================================================
# 示例 5: FID评价
# ============================================================================

def example_fid():
    """FID评价示例"""
    print("\n" + "="*80)
    print("示例 5: FID评价")
    print("="*80)
    
    print("\nFID（Fréchet Inception Distance）只支持多样本评价")
    print("需要两个目录：一个包含生成图像，一个包含真实图像")
    print("\n示例代码：")
    print("""
    from evaluation import calculate_fid_score
    
    # 计算FID
    fid_value = calculate_fid_score(
        path_fake='path/to/generated/images',
        path_real='path/to/real/images'
    )
    
    print(f"FID Score: {fid_value:.2f}")
    """)


# ============================================================================
# 示例 6: 完整工作流程
# ============================================================================

def example_complete_workflow():
    """完整工作流程示例"""
    print("\n" + "="*80)
    print("示例 6: 完整工作流程")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    evaluator = BatchEvaluator(device=device)
    
    # 1. 准备数据
    n_samples = 3
    print(f"\n1. 生成 {n_samples} 对测试图像...")
    img_list1 = [np.random.rand(256, 256, 3) for _ in range(n_samples)]
    img_list2 = [np.random.rand(256, 256, 3) for _ in range(n_samples)]
    
    # 2. 计算标准指标
    print(f"2. 计算标准评价指标...")
    results = evaluator.evaluate_batch(
        img_list1, 
        img_list2,
        metrics=['L1', 'L2', 'RMSE', 'PSNR', 'SSIM', 'LPIPS']
    )
    
    print("\n标准指标结果：")
    for metric, stats in results.items():
        print(f"\n{metric}:")
        print(f"  Mean ± Std: {stats['mean']:.6f} ± {stats['std']:.6f}")
        print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
    
    # 3. FID计算
    print("\n3. FID需要实际的图像目录（此处跳过演示）")
    print("   如需计算FID，使用:")
    print("   fid = calculate_fid_score('path/fake', 'path/real')")


# ============================================================================
# 实用指南
# ============================================================================

def print_usage_guide():
    """打印使用指南"""
    guide = """
╔════════════════════════════════════════════════════════════════════════════╗
║                        评价模块使用指南                                    ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 支持的指标：
  ✓ L1:     L1距离 (平均绝对误差)
  ✓ L2:     L2距离 (平均平方误差)
  ✓ RMSE:   均方根误差
  ✓ PSNR:   峰值信噪比 (dB)
  ✓ SSIM:   结构相似性指数
  ✓ LPIPS:  学习感知图像块相似性
  ✓ FID:    Fréchet Inception Distance (仅多样本)

🔧 快速开始：

1️⃣  单样本评价：
    from evaluation import evaluate_single_sample
    
    results = evaluate_single_sample(img1, img2)
    print(f"L1: {results['L1']:.6f}")

2️⃣  多样本评价：
    from evaluation import evaluate_batch_samples
    
    results = evaluate_batch_samples(img_list1, img_list2)
    print(f"PSNR Mean: {results['PSNR']['mean']:.2f}")

3️⃣  FID计算：
    from evaluation import calculate_fid_score
    
    fid = calculate_fid_score('path/fake', 'path/real')
    print(f"FID: {fid:.2f}")

📋 输入格式支持：
  ✓ NumPy数组: (H,W), (H,W,C), (C,H,W)
  ✓ PyTorch张量: (H,W), (C,H,W), (B,C,H,W)
  ✓ 值范围: [0,1] 或 [0,255] (自动检测)

⚙️  依赖包：
  - torch, torchvision: 基础
  - scikit-image: SSIM
  - lpips: LPIPS
  - pytorch_fid: FID

💡 提示：
  • 所有操作自动支持GPU加速
  • 指定metrics参数可计算子集指标
  • 批量评价返回均值、标准差、最小值、最大值
  • FID只支持多样本，需要两个图像目录
    """
    print(guide)


if __name__ == "__main__":
    print_usage_guide()
    
    # 运行示例
    example_single_sample()
    example_batch_evaluation()
    example_convenience_functions()
    example_with_image_files()
    example_fid()
    example_complete_workflow()
    
    print("\n" + "="*80)
    print("✅ 所有示例演示完成！")
    print("="*80)
