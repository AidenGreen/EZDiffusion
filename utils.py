# utils.py
import os
import random
from datetime import datetime

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def set_seed(seed: int):
    """
    固定 Python、NumPy、PyTorch 的随机种子，方便复现实验结果。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def denormalize(x):
    """
    将图像从 [-1, 1] 还原到 [0, 1]，用于保存和 TensorBoard 可视化。
    """
    return (x + 1.0) / 2.0


def build_model_input(noisy_images, conditions=None):
    """
    将 noisy_images 和 conditions 拼接成扩散模型输入。
    """
    if conditions is None:
        return noisy_images

    return torch.cat([noisy_images, conditions], dim=1)


def frequency_lowpass_condition(images, keep_rate=0.25):
    """
    使用高斯频域低通生成低频引导图，输入输出均为 [B, C, H, W]。
    """
    if images.ndim != 4:
        raise ValueError(
            f"frequency_lowpass_condition 只支持 [B, C, H, W]，当前 shape={images.shape}"
        )

    keep_rate = float(max(min(keep_rate, 1.0), 1e-4))

    x = images.float()
    _, _, h, w = x.shape

    fy = torch.linspace(
        -1.0,
        1.0,
        h,
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, h, 1)

    fx = torch.linspace(
        -1.0,
        1.0,
        w,
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 1, w)

    radius_square = fx * fx + fy * fy

    gaussian_mask = torch.exp(
        -0.5 * radius_square / (keep_rate * keep_rate)
    )

    freq = torch.fft.fft2(x, dim=(-2, -1))
    freq = torch.fft.fftshift(freq, dim=(-2, -1))
    freq = freq * gaussian_mask
    freq = torch.fft.ifftshift(freq, dim=(-2, -1))

    lowpass = torch.fft.ifft2(freq, dim=(-2, -1)).real
    lowpass = lowpass.clamp(-1.0, 1.0)

    return lowpass.to(images.dtype)


def create_run_dirs(runs_dir, experiment_name):
    """
    根据实验名和秒级时间戳创建 run、checkpoints、samples 三个目录。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_name}_{timestamp}"

    run_dir = os.path.join(runs_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    sample_dir = os.path.join(run_dir, "samples")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    return run_dir, checkpoint_dir, sample_dir


def find_latest_run_dir(runs_dir, experiment_name):
    """
    从 runs 目录中查找当前实验名对应的最新 run 目录。
    """
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"runs 目录不存在: {runs_dir}")

    prefix = f"{experiment_name}_"

    run_dirs = [
        os.path.join(runs_dir, name)
        for name in os.listdir(runs_dir)
        if name.startswith(prefix) and os.path.isdir(os.path.join(runs_dir, name))
    ]

    if len(run_dirs) == 0:
        raise FileNotFoundError(f"没有找到实验 {experiment_name} 对应的 run 目录。")

    run_dirs = sorted(run_dirs, key=os.path.getmtime, reverse=True)

    return run_dirs[0]


def save_sample_grid(images, save_path, nrow=4):
    """
    将一批生成图像保存为网格图，并返回 TensorBoard 可写入的 grid。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    images = denormalize(images).clamp(0.0, 1.0)

    grid = make_grid(
        images,
        nrow=nrow,
        padding=2,
    )

    save_image(grid, save_path)

    return grid