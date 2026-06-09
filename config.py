# config.py
import os
import torch


# ============================================================
# Project paths
# ============================================================

# 当前实验名称；训练开始时会拼接秒级时间戳生成唯一 run 目录。
EXPERIMENT_NAME = "diff_demo_cifar10"

# 项目根目录，也就是当前 config.py 所在目录。
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 本地第三方库目录，用于放 diffusers 源码。
THIRDPARTY_DIR = os.path.join(PROJECT_ROOT, "_thirdparty")

# 本地 diffusers 仓库目录。
DIFFUSERS_DIR = os.path.join(THIRDPARTY_DIR, "diffusers")

# diffusers 真正的源码目录。
DIFFUSERS_SRC_DIR = os.path.join(DIFFUSERS_DIR, "src")

# 数据集目录，CIFAR-10 会下载到这里。
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# 所有实验 run 的根目录。
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

# 测试时指定 run 目录；None 表示自动选择当前 EXPERIMENT_NAME 下最新的 run。
TEST_RUN_DIR = None

# 自动创建必要目录。
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)


# ============================================================
# Dataset config
# ============================================================

# 数据集名称；当前支持 "cifar10" 和 "fake"。
DATASET_NAME = "cifar10"

# 输入图像尺寸；CIFAR-10 原始尺寸是 32。
IMAGE_SIZE = 32

# 输入图像通道数；CIFAR-10 是 RGB，所以是 3。
IMAGE_CHANNELS = 3

# 条件通道数；普通 DDPM 为 0，后期条件扩散时再改。
CONDITION_CHANNELS = 0

# 训练 batch size。
BATCH_SIZE = 128

# DataLoader 进程数；Windows 下调试建议 0，稳定后可改 2 或 4。
NUM_WORKERS = 0

# 是否启用 DataLoader pin_memory；CUDA 训练时建议 True。
PIN_MEMORY = True


# ============================================================
# Training config
# ============================================================

# 训练总 epoch 数。
EPOCHS = 50

# AdamW 学习率。
LEARNING_RATE = 1e-4

# AdamW 权重衰减。
WEIGHT_DECAY = 1e-4

# 梯度裁剪阈值；设为 None 或 0 表示不裁剪。
GRAD_CLIP_NORM = 1.0

# 随机种子。
SEED = 777

# 打印训练日志、写入 TensorBoard 标量的 step 间隔。
PRINT_EVERY_STEPS = 50

# 保存 latest.pt 的 step 间隔。
SAVE_EVERY_STEPS = 500

# 测试集评估 step 间隔；触发时会计算 test loss，并保存 top-k 最优权重。
TEST_EVERY_STEPS = 500

# 每次测试最多跑多少个 test batch，避免测试太慢。
TEST_MAX_BATCHES = 20

# 生成 sample 图像的 step 间隔；触发时写 TensorBoard 并保存到 samples。
SAMPLE_EVERY_STEPS = 1000

# 保存 test loss 最好的前 K 组权重。
SAVE_BEST_K = 3


# ============================================================
# Diffusion config
# ============================================================

# 训练时扩散总步数。
NUM_TRAIN_TIMESTEPS = 1000

# 采样时反推步数；调试时 100 快，正式可以改 1000。
NUM_INFERENCE_STEPS = 100

# beta schedule 类型；DDPMScheduler 常用 "linear"。
BETA_SCHEDULE = "linear"

# 模型预测目标；可选 "epsilon"、"sample"、"v_prediction"。
PREDICTION_TYPE = "epsilon"


# ============================================================
# UNet config
# ============================================================

# UNet 每个 stage 的通道数。
UNET_BLOCK_OUT_CHANNELS = (64, 128, 256)

# UNet 下采样模块类型。
UNET_DOWN_BLOCK_TYPES = (
    "DownBlock2D",
    "AttnDownBlock2D",
    "DownBlock2D",
)

# UNet 上采样模块类型。
UNET_UP_BLOCK_TYPES = (
    "UpBlock2D",
    "AttnUpBlock2D",
    "UpBlock2D",
)

# 每个 block 内部 ResNet 层数。
LAYERS_PER_BLOCK = 2


# ============================================================
# Runtime config
# ============================================================

# 训练设备，优先使用 CUDA。
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 是否启用 AMP 混合精度。
USE_AMP = True

# AMP 数据类型；3090/4090 常用 float16。
AMP_DTYPE = torch.float16


# ============================================================
# TensorBoard config
# ============================================================

# 是否启用 TensorBoard。
USE_TENSORBOARD = True

# TensorBoard sample 图像每次生成多少张。
TB_SAMPLE_BATCH_SIZE = 16

# TensorBoard sample 图像网格每行多少张。
TB_SAMPLE_NROW = 4
