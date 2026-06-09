# model.py
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from utils import build_model_input

from _thirdparty.diffusers.models.unets.unet_2d import UNet2DModel
from _thirdparty.diffusers.schedulers.scheduling_ddpm import DDPMScheduler


class DiffusionModel(nn.Module):
    """
    扩散模型封装类，负责 loss 计算、训练一步、测试一步、采样和权重读写。
    """

    def __init__(self):
        """
        初始化 UNet、DDPM scheduler、optimizer 和 AMP scaler。
        """
        super().__init__()

        self.device_name = config.DEVICE
        self.image_channels = config.IMAGE_CHANNELS
        self.condition_channels = config.CONDITION_CHANNELS

        self.unet = UNet2DModel(
            sample_size=config.IMAGE_SIZE,
            in_channels=config.IMAGE_CHANNELS + config.CONDITION_CHANNELS,
            out_channels=config.IMAGE_CHANNELS,
            layers_per_block=config.LAYERS_PER_BLOCK,
            block_out_channels=config.UNET_BLOCK_OUT_CHANNELS,
            down_block_types=config.UNET_DOWN_BLOCK_TYPES,
            up_block_types=config.UNET_UP_BLOCK_TYPES,
        )

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.NUM_TRAIN_TIMESTEPS,
            beta_schedule=config.BETA_SCHEDULE,
            prediction_type=config.PREDICTION_TYPE,
        )

        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.use_amp = bool(config.USE_AMP and config.DEVICE == "cuda")

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp,
        )

        self.to(self.device_name)

    def get_autocast_context(self):
        """
        返回 AMP 混合精度上下文；未启用 AMP 时返回空上下文。
        """
        if self.use_amp:
            return torch.amp.autocast(
                "cuda",
                enabled=True,
                dtype=config.AMP_DTYPE,
            )

        return nullcontext()

    def compute_diffusion_loss(self, clean_images, conditions=None):
        """
        根据 prediction_type 计算 epsilon、sample 或 v_prediction 训练损失。
        """
        batch_size = clean_images.shape[0]

        noise = torch.randn_like(clean_images)

        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(batch_size,),
            device=clean_images.device,
            dtype=torch.long,
        )

        noisy_images = self.noise_scheduler.add_noise(
            original_samples=clean_images,
            noise=noise,
            timesteps=timesteps,
        )

        model_input = build_model_input(
            noisy_images=noisy_images,
            conditions=conditions,
        )

        model_pred = self.unet(
            sample=model_input,
            timestep=timesteps,
        ).sample

        prediction_type = self.noise_scheduler.config.prediction_type

        if prediction_type == "epsilon":
            target = noise

        elif prediction_type == "sample":
            target = clean_images

        elif prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                sample=clean_images,
                noise=noise,
                timesteps=timesteps,
            )

        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")

        loss = F.mse_loss(model_pred, target)

        return loss

    def train_one_step(self, clean_images, conditions=None):
        """
        执行一次训练更新，并返回当前 step 的 loss 指标。
        """
        self.train()
        self.optimizer.zero_grad(set_to_none=True)

        with self.get_autocast_context():
            loss = self.compute_diffusion_loss(
                clean_images=clean_images,
                conditions=conditions,
            )

        self.scaler.scale(loss).backward()

        if config.GRAD_CLIP_NORM is not None and config.GRAD_CLIP_NORM > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.unet.parameters(),
                max_norm=config.GRAD_CLIP_NORM,
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {
            "loss": float(loss.detach().cpu().item())
        }

    @torch.no_grad()
    def eval_one_step(self, clean_images, conditions=None):
        """
        执行一次测试前向计算，只返回 loss，不更新参数。
        """
        self.eval()

        with self.get_autocast_context():
            loss = self.compute_diffusion_loss(
                clean_images=clean_images,
                conditions=conditions,
            )

        return {
            "loss": float(loss.detach().cpu().item())
        }

    @torch.no_grad()
    def sample(self, batch_size=16, conditions=None):
        """
        从随机噪声开始反向采样，生成一批图像。
        """
        self.eval()

        image = torch.randn(
            batch_size,
            self.image_channels,
            config.IMAGE_SIZE,
            config.IMAGE_SIZE,
            device=self.device_name,
        )

        if conditions is not None:
            conditions = conditions.to(self.device_name).float()

        self.noise_scheduler.set_timesteps(
            num_inference_steps=config.NUM_INFERENCE_STEPS,
            device=self.device_name,
        )

        for t in self.noise_scheduler.timesteps:
            model_input = build_model_input(
                noisy_images=image,
                conditions=conditions,
            )

            model_pred = self.unet(
                sample=model_input,
                timestep=t,
            ).sample

            image = self.noise_scheduler.step(
                model_output=model_pred,
                timestep=t,
                sample=image,
            ).prev_sample

        return image.clamp(-1.0, 1.0)

    def save_checkpoint(self, path, step=0, epoch=0, extra=None):
        """
        保存模型、优化器、AMP scaler 和关键训练配置。
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                "step": step,
                "epoch": epoch,
                "unet": self.unet.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "extra": extra or {},
                "config": {
                    "image_size": config.IMAGE_SIZE,
                    "image_channels": config.IMAGE_CHANNELS,
                    "condition_channels": config.CONDITION_CHANNELS,
                    "num_train_timesteps": config.NUM_TRAIN_TIMESTEPS,
                    "num_inference_steps": config.NUM_INFERENCE_STEPS,
                    "beta_schedule": config.BETA_SCHEDULE,
                    "prediction_type": config.PREDICTION_TYPE,
                },
            },
            path,
        )

    def load_checkpoint(self, path, load_optimizer=True):
        """
        加载 checkpoint，并按需恢复 optimizer 和 AMP scaler。
        """
        checkpoint = torch.load(
            path,
            map_location=self.device_name,
        )

        self.unet.load_state_dict(checkpoint["unet"])

        if load_optimizer:
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            if "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler"])

        return {
            "step": checkpoint.get("step", 0),
            "epoch": checkpoint.get("epoch", 0),
            "extra": checkpoint.get("extra", {}),
            "config": checkpoint.get("config", {}),
        }