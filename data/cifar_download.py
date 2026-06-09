# data/Tool_download.py
import os
from torchvision.datasets import CIFAR10


DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    """
    下载 CIFAR-10 到当前 data 文件夹。
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    print("===================================")
    print("Download CIFAR-10")
    print("===================================")
    print(f"Target dir: {DATA_DIR}")

    train_set = CIFAR10(
        root=DATA_DIR,
        train=True,
        download=True,
    )

    test_set = CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
    )

    print("===================================")
    print("CIFAR-10 download finished.")
    print(f"Train images: {len(train_set)}")
    print(f"Test images : {len(test_set)}")
    print("===================================")


if __name__ == "__main__":
    main()