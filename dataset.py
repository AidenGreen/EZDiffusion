# dataset.py
import os

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, FakeData

import config


def build_image_transform():
    """
    构建图像读取变换：Resize、ToTensor、Normalize 到 [-1, 1]。
    """
    if config.IMAGE_CHANNELS == 1:
        mean = [0.5]
        std = [0.5]
    elif config.IMAGE_CHANNELS == 3:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        raise ValueError("当前 dataset.py 只支持 IMAGE_CHANNELS = 1 或 3。")

    transform_list = []

    if config.IMAGE_SIZE != 32:
        transform_list.append(
            transforms.Resize(
                (config.IMAGE_SIZE, config.IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
        )

    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    return transforms.Compose(transform_list)


def build_dataset(train=True):
    """
    根据 config.DATASET_NAME 构建训练集或测试集。
    """
    transform = build_image_transform()
    dataset_name = config.DATASET_NAME.lower()

    if dataset_name == "cifar10":
        cifar_dir = os.path.join(config.DATA_DIR, "cifar-10-batches-py")

        if not os.path.isdir(cifar_dir):
            raise FileNotFoundError(
                "没有找到 CIFAR-10 数据集。\n"
                f"期望路径：{cifar_dir}\n"
                "请先运行：python data\\Tool_download.py"
            )

        return CIFAR10(
            root=config.DATA_DIR,
            train=train,
            download=False,
            transform=transform,
        )

    if dataset_name == "fake":
        return FakeData(
            size=4096 if train else 512,
            image_size=(config.IMAGE_CHANNELS, config.IMAGE_SIZE, config.IMAGE_SIZE),
            num_classes=10,
            transform=transform,
        )

    raise ValueError(f"未知数据集: {config.DATASET_NAME}")


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