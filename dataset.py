# dataset.py
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import config

normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)

class APDataset(Dataset):
    def __init__(self, data_root, train=True) -> None:
        super().__init__()

        self.data_root = data_root

        self.gt_path = os.path.join(data_root, "absolute_phase_normal.npy")

        self.data = [
            os.path.join(data_root, "absolute_phase_glass.npy"),
            # os.path.join(data_root, "absolute_phase_metal.npy"),
        ]

    def __len__(self):
        return len(self.data)

    def resize_phase(self, phase):
        """
        phase: torch.Tensor, shape = [1, H, W]
        return: shape = [1, IMAGE_SIZE, IMAGE_SIZE]
        """
        if phase.shape[-2:] == (config.IMAGE_SIZE, config.IMAGE_SIZE):
            return phase

        phase = phase.unsqueeze(0)  # [1, 1, H, W]

        phase = F.interpolate(
            phase,
            size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        phase = phase.squeeze(0)  # [1, IMAGE_SIZE, IMAGE_SIZE]

        return phase

    def __getitem__(self, index):
        data_path = self.data[index]

        input_phase = np.load(data_path).astype(np.float32)
        gt_phase = np.load(self.gt_path).astype(np.float32)

        # NaN 替换为 0，不做其他处理
        input_phase = np.where(np.isnan(input_phase), 0.0, input_phase)
        gt_phase = np.where(np.isnan(gt_phase), 0.0, gt_phase)

        input_phase = np.squeeze(input_phase)
        gt_phase = np.squeeze(gt_phase)

        input_phase = torch.from_numpy(input_phase).float().unsqueeze(0)
        gt_phase = torch.from_numpy(gt_phase).float().unsqueeze(0)

        input_phase = self.resize_phase(input_phase)
        gt_phase = self.resize_phase(gt_phase)

        return normalize(input_phase), normalize(gt_phase)


def build_dataset(train=True):
    """
    根据 config.DATASET_NAME 构建训练集或测试集。
    """
    dataset = APDataset("./data/SimDatas/", train=train)
    return dataset


def build_train_dataloader():
    """
    构建训练 DataLoader，只负责加载数据。
    """
    return DataLoader(
        build_dataset(train=True),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.PIN_MEMORY and config.DEVICE == "cuda"),
        drop_last=True,
    )


def build_test_dataloader():
    """
    构建测试 DataLoader，只负责加载数据。
    """
    return DataLoader(
        build_dataset(train=False),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.PIN_MEMORY and config.DEVICE == "cuda"),
        drop_last=False,
    )