# FontDiffuser

基于扩散模型的字体风格迁移项目：输入内容字符（或内容图）和风格图，生成具有目标风格的字体图像。项目包含边缘条件注入、命令行推理、Gradio WebUI、训练和图像质量评价功能。

## 环境

建议使用 Python 3.10+ 和 PyTorch（CUDA 环境可显著提升推理与训练速度）。

```bash
conda create -n fontdiffuser python=3.10
conda activate fontdiffuser
pip install -r requirements.txt
```

请根据本机 CUDA 版本安装匹配的 PyTorch 与 torchvision。

## 快速推理

准备模型权重、风格图和字体文件后运行：

```bash
python sample.py \
	--ckpt_dir ckpt/origin \
	--style_image_path path/to/style.png \
	--character_input \
	--content_character 隆 \
	--ttf_path ttf/KaiXinSongA.ttf \
	--save_image \
	--save_image_dir outputs/sample \
	--device auto
```

也可以使用内容图，将 `--character_input` 和 `--content_character` 替换为：

```bash
--content_image_path path/to/content.png
```

## WebUI

```bash
python gradio_app.py --device auto
```

启动后可进行单样本生成、模型对比和批量评价。

## 训练

训练数据目录格式：

```text
data_root/
├── train/
│   ├── ContentImage/<character>.jpg
│   └── TargetImage/<style>/<style>+<character>.jpg
└── val/
		├── ContentImage/<character>.jpg
		└── TargetImage/<style>/<style>+<character>.jpg
```

```bash
accelerate launch train.py \
	--data_root data_examples \
	--output_dir outputs/FontDiffuser
```


## 主要入口

- `sample.py`：命令行推理
- `gradio_app.py`：交互式 WebUI
- `batch_webui.py`：批量推理与评价 WebUI
- `train.py`：模型训练
- `evaluation.py`：L1、L2、RMSE、PSNR、SSIM、LPIPS 和 FID 评价