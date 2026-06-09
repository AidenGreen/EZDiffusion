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
    根据是否存在条件图，将 noisy_images 和 conditions 拼接成模型输入。
    """
    if conditions is None:
        return noisy_images

    return torch.cat([noisy_images, conditions], dim=1)


def create_run_dirs(runs_dir, experiment_name):
    """
    根据实验名和秒级时间戳创建 run、checkpoints、samples 三个目录。
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_name}_{timestamp[2:]}"

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
        raise FileNotFoundError(
            f"没有找到实验 {experiment_name} 对应的 run 目录。"
        )

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