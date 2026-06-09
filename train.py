# train.py
import os
import time

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import build_train_dataloader, build_test_dataloader, preprocess_batch
from model import DiffusionModel
from utils import set_seed, save_sample_grid, create_run_dirs


def main():
    """
    主训练入口：为本次实验创建独立 run 目录，训练模型、测试、采样和保存权重。
    """
    set_seed(config.SEED)

    run_dir, checkpoint_dir, sample_dir = create_run_dirs(
        runs_dir=config.RUNS_DIR,
        experiment_name=config.EXPERIMENT_NAME,
    )

    latest_ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
    best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")

    print("===================================")
    print("EZDiffusion Training")
    print("===================================")
    print(f"Experiment     : {config.EXPERIMENT_NAME}")
    print(f"Run dir        : {run_dir}")
    print(f"Checkpoints    : {checkpoint_dir}")
    print(f"Samples        : {sample_dir}")
    print(f"Dataset        : {config.DATASET_NAME}")
    print(f"Data dir       : {config.DATA_DIR}")
    print(f"Image size     : {config.IMAGE_SIZE}")
    print(f"Channels       : {config.IMAGE_CHANNELS}")
    print(f"Condition      : {config.CONDITION_CHANNELS}")
    print(f"Prediction     : {config.PREDICTION_TYPE}")
    print(f"Device         : {config.DEVICE}")
    print(f"AMP            : {config.USE_AMP}")
    print("===================================")

    train_loader = build_train_dataloader()
    test_loader = build_test_dataloader()

    model = DiffusionModel()
    writer = SummaryWriter(run_dir) if config.USE_TENSORBOARD else None

    global_step = 0
    best_test_loss = float("inf")
    best_test_records = []
    start_time = time.time()

    for epoch in range(config.EPOCHS):
        epoch_loss_sum = 0.0
        epoch_step_count = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config.EPOCHS}",
        )

        for raw_batch in progress_bar:
            clean_images, conditions = preprocess_batch(
                raw_batch=raw_batch,
                device=config.DEVICE,
            )

            log_dict = model.train_one_step(
                clean_images=clean_images,
                conditions=conditions,
            )

            loss = log_dict["loss"]
            epoch_loss_sum += loss
            epoch_step_count += 1

            progress_bar.set_postfix(
                {
                    "step": global_step,
                    "loss": f"{loss:.4f}",
                    "best_test": f"{best_test_loss:.4f}",
                }
            )

            if global_step % config.TEST_EVERY_STEPS == 0:
                test_loss_sum = 0.0
                test_count = 0

                for test_i, raw_test_batch in enumerate(test_loader):
                    if test_i >= config.TEST_MAX_BATCHES:
                        break

                    test_images, test_conditions = preprocess_batch(
                        raw_batch=raw_test_batch,
                        device=config.DEVICE,
                    )

                    test_log = model.eval_one_step(
                        clean_images=test_images,
                        conditions=test_conditions,
                    )

                    test_loss_sum += test_log["loss"]
                    test_count += 1

                test_loss = test_loss_sum / max(test_count, 1)

                print(
                    f"\n[Test] "
                    f"step={global_step}, "
                    f"test_loss={test_loss:.6f}"
                )

                if writer is not None:
                    writer.add_scalar("test/test_loss", test_loss, global_step)

                ckpt_name = f"best_step_{global_step:08d}_loss_{test_loss:.6f}.pt"
                ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

                model.save_checkpoint(
                    path=ckpt_path,
                    step=global_step,
                    epoch=epoch,
                    extra={
                        "test_loss": test_loss,
                        "run_dir": run_dir,
                    },
                )

                best_test_records.append((test_loss, ckpt_path))
                best_test_records = sorted(best_test_records, key=lambda x: x[0])

                if test_loss < best_test_loss:
                    best_test_loss = test_loss

                    model.save_checkpoint(
                        path=best_ckpt_path,
                        step=global_step,
                        epoch=epoch,
                        extra={
                            "test_loss": test_loss,
                            "run_dir": run_dir,
                        },
                    )

                while len(best_test_records) > config.SAVE_BEST_K:
                    _, remove_path = best_test_records.pop(-1)
                    if os.path.exists(remove_path):
                        os.remove(remove_path)

                model.train()

            if global_step % config.SAMPLE_EVERY_STEPS == 0:
                samples = model.sample(
                    batch_size=config.TB_SAMPLE_BATCH_SIZE,
                )

                sample_path = os.path.join(
                    sample_dir,
                    f"sample_step_{global_step:08d}.png",
                )

                grid = save_sample_grid(
                    images=samples,
                    save_path=sample_path,
                    nrow=config.TB_SAMPLE_NROW,
                )

                print(f"\n[Sample] saved: {sample_path}")

                if writer is not None:
                    writer.add_image("test/generated", grid, global_step)

                model.train()

            if global_step % config.SAVE_EVERY_STEPS == 0:
                model.save_checkpoint(
                    path=latest_ckpt_path,
                    step=global_step,
                    epoch=epoch,
                    extra={
                        "run_dir": run_dir,
                    },
                )
                print(f"\n[Save] {latest_ckpt_path}")

            if global_step % config.PRINT_EVERY_STEPS == 0:
                elapsed = time.time() - start_time

                print(
                    f"\n[Train] "
                    f"epoch={epoch + 1}, "
                    f"step={global_step}, "
                    f"loss={loss:.4f}, "
                    f"best_test={best_test_loss:.4f}, "
                    f"elapsed={elapsed:.1f}s"
                )

                if writer is not None:
                    writer.add_scalar("training/train_step", loss, global_step)
                    writer.add_scalar("training/lr", model.optimizer.param_groups[0]["lr"], global_step)

            global_step += 1

        epoch_loss = epoch_loss_sum / max(epoch_step_count, 1)

        print(
            f"\n[Epoch] "
            f"epoch={epoch + 1}, "
            f"epoch_loss={epoch_loss:.6f}"
        )

        if writer is not None:
            writer.add_scalar("training/train_epoch", epoch_loss, epoch + 1)

        model.save_checkpoint(
            path=latest_ckpt_path,
            step=global_step,
            epoch=epoch,
            extra={
                "epoch_loss": epoch_loss,
                "run_dir": run_dir,
            },
        )

    if writer is not None:
        writer.close()

    print("===================================")
    print("Training finished.")
    print(f"Run dir         : {run_dir}")
    print(f"Latest ckpt     : {latest_ckpt_path}")
    print(f"Best ckpt       : {best_ckpt_path}")
    print("===================================")


if __name__ == "__main__":
    main()