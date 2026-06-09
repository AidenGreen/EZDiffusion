# test.py
import os

import config
from model import DiffusionModel
from utils import set_seed, save_sample_grid


def main():
    """
    测试入口：加载 checkpoint 并生成一组 sample 图像。
    """
    set_seed(config.SEED)

    ckpt_path = config.BEST_CKPT_PATH

    if not os.path.exists(ckpt_path):
        ckpt_path = config.LAST_CKPT_PATH

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("没有找到 checkpoint，请先运行 train.py。")

    print("===================================")
    print("EZDiffusion Sampling")
    print("===================================")
    print(f"Checkpoint      : {ckpt_path}")
    print(f"Prediction type : {config.PREDICTION_TYPE}")
    print(f"Device          : {config.DEVICE}")
    print(f"Image size      : {config.IMAGE_SIZE}")
    print(f"Inference steps : {config.NUM_INFERENCE_STEPS}")
    print("===================================")

    model = DiffusionModel()

    info = model.load_checkpoint(
        path=ckpt_path,
        load_optimizer=False,
    )

    print(f"Loaded checkpoint: epoch={info['epoch']}, step={info['step']}")

    samples = model.sample(
        batch_size=config.TB_SAMPLE_BATCH_SIZE,
    )

    save_path = os.path.join(config.SAMPLE_DIR, "sample_grid.png")

    save_sample_grid(
        images=samples,
        save_path=save_path,
        nrow=config.TB_SAMPLE_NROW,
    )

    print(f"Saved sample grid: {save_path}")


if __name__ == "__main__":
    main()