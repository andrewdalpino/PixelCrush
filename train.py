import random

from argparse import ArgumentParser
from functools import partial

import torch

from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.amp import autocast
from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.backends.mps import is_available as mps_is_available
from torch.utils.tensorboard import SummaryWriter

from torchvision.transforms.v2 import CenterCrop, RandomCrop

from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from data import ImageFolder
from model import PixelCrush

from tqdm import tqdm


def main():
    parser = ArgumentParser(description="Training script")

    parser.add_argument("--train_images_path", default="./dataset/train", type=str)
    parser.add_argument("--test_images_path", default="./dataset/test", type=str)
    parser.add_argument("--num_dataset_processes", default=2, type=int)
    parser.add_argument(
        "--downscale_ratio",
        default=0.5,
        type=int,
        choices=PixelCrush.AVAILABLE_DOWNSCALE_RATIOS,
    )
    parser.add_argument("--target_resolution", default=256, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--max_gradient_norm", default=2.0, type=float)
    parser.add_argument("--num_channels", default=32, type=int)
    parser.add_argument("--hidden_ratio", default=1, type=int)
    parser.add_argument("--num_encoder_layers", default=20, type=int)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--eval_interval", default=2, type=int)
    parser.add_argument("--checkpoint_interval", default=2, type=int)
    parser.add_argument(
        "--checkpoint_path", default="./checkpoints/checkpoint.pt", type=str
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_dir_path", default="./runs", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=None, type=int)

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(f"Batch size must be greater than 0, {args.batch_size} given.")

    if args.learning_rate < 0:
        raise ValueError(
            f"Learning rate must be a positive value, {args.learning_rate} given."
        )

    if args.num_epochs < 1:
        raise ValueError(f"Must train for at least 1 epoch, {args.num_epochs} given.")

    if args.eval_interval < 1:
        raise ValueError(
            f"Eval interval must be greater than 0, {args.eval_interval} given."
        )

    if args.checkpoint_interval < 1:
        raise ValueError(
            f"Checkpoint interval must be greater than 0, {args.checkpoint_interval} given."
        )

    if "cuda" in args.device and not cuda_is_available():
        raise RuntimeError("Cuda is not available.")

    if "mps" in args.device and not mps_is_available():
        raise RuntimeError("MPS is not available.")

    torch.set_float32_matmul_precision("high")

    dtype = (
        torch.bfloat16
        if ("cuda" in args.device and is_bf16_supported()) or args.device == "mps"
        else torch.float32
    )

    amp_context = autocast(device_type=args.device, dtype=dtype)

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    logger = SummaryWriter(args.run_dir_path)

    new_dataset = partial(
        ImageFolder,
        target_resolution=args.target_resolution,
        downscale_ratio=args.downscale_ratio,
    )

    training = new_dataset(
        args.train_images_path,
        pre_transformer=RandomCrop(args.target_resolution),
    )

    testing = new_dataset(
        args.test_images_path,
        pre_transformer=CenterCrop(args.target_resolution),
    )

    new_dataloader = partial(
        DataLoader,
        batch_size=args.batch_size,
        pin_memory="cuda" in args.device,
        num_workers=args.num_dataset_processes,
    )

    train_loader = new_dataloader(training, shuffle=True)
    test_loader = new_dataloader(testing)

    model_args = {
        "downscale_ratio": args.downscale_ratio,
        "num_channels": args.num_channels,
        "hidden_ratio": args.hidden_ratio,
        "num_encoder_layers": args.num_encoder_layers,
    }

    model = PixelCrush(**model_args)

    model.add_weight_norms()

    if args.activation_checkpointing:
        model.encoder.enable_activation_checkpointing()

    model = model.to(args.device)

    l2_loss_function = MSELoss()
    bicubic_l1_loss_function = L1Loss()

    if "cuda" in args.device:
        print("Compiling models")

        model = torch.compile(model)

    print(f"Downscaler has {model.num_trainable_params:,} trainable parameters")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    y_pred_psnr_metric = PeakSignalNoiseRatio().to(args.device)
    y_pred_ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)

    y_bicubic_psnr_metric = PeakSignalNoiseRatio().to(args.device)
    y_bicubic_ssim_metric = StructuralSimilarityIndexMeasure().to(args.device)

    starting_epoch = 1

    if args.resume:
        checkpoint = torch.load(
            args.checkpoint_path, map_location=args.device, weights_only=True
        )

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        starting_epoch += checkpoint["epoch"]

        print("Previous checkpoint resumed successfully")

    print("Training ...")
    model.train()

    for epoch in range(starting_epoch, args.num_epochs + 1):
        total_l2_loss, total_bicubic_l1 = 0.0, 0.0
        total_batches, total_steps = 0, 0
        total_gradient_norm = 0.0

        for step, (x, y) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False), start=1
        ):
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)

            with amp_context:
                y_pred, y_bicubic = model.forward(x)

                l2_loss = l2_loss_function(y_pred, y)

                scaled_loss = l2_loss / args.gradient_accumulation_steps

            scaled_loss.backward()

            if step % args.gradient_accumulation_steps == 0:
                norm = clip_grad_norm_(model.parameters(), args.max_gradient_norm)

                optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                total_gradient_norm += norm.item()

                total_steps += 1

            with torch.no_grad():
                bicubic_l1 = bicubic_l1_loss_function(y_bicubic, y)

            total_l2_loss += l2_loss.item()

            total_batches += 1

        average_l2_loss = total_l2_loss / total_batches
        average_gradient_norm = total_gradient_norm / total_steps
        average_bicubic_l1 = total_bicubic_l1 / total_batches

        logger.add_scalar("Pixel L2", average_l2_loss, epoch)
        logger.add_scalar("Bicubic L1", average_bicubic_l1, epoch)
        logger.add_scalar("Gradient Norm", average_gradient_norm, epoch)

        print(
            f"Epoch {epoch}:",
            f"Pixel L2: {average_l2_loss:.4},",
            f"Bicubic L1: {average_bicubic_l1:.4}",
            f"Gradient Norm: {average_gradient_norm:.4},",
        )

        if epoch % args.eval_interval == 0:
            model.eval()

            for x, y in tqdm(test_loader, desc="Testing", leave=False):
                x = x.to(args.device, non_blocking=True)
                y = y.to(args.device, non_blocking=True)

                y_pred, y_bicubic = model.test_compare(x)

                y_pred_psnr_metric.update(y_pred, y)
                y_pred_ssim_metric.update(y_pred, y)

                y_bicubic_psnr_metric.update(y_bicubic, y)
                y_bicubic_ssim_metric.update(y_bicubic, y)

            y_pred_psnr = y_pred_psnr_metric.compute()
            y_pred_ssim = y_pred_ssim_metric.compute()

            y_bicubic_psnr = y_bicubic_psnr_metric.compute()
            y_bicubic_ssim = y_bicubic_ssim_metric.compute()

            logger.add_scalar("Enhanced PSNR", y_pred_psnr, epoch)
            logger.add_scalar("Enhanced SSIM", y_pred_ssim, epoch)

            logger.add_scalar("Bicubic PSNR", y_bicubic_psnr, epoch)
            logger.add_scalar("Bicubic SSIM", y_bicubic_ssim, epoch)

            print(
                f"PSNR: {y_bicubic_psnr:.5} / {y_pred_psnr:.5},",
                f"SSIM: {y_bicubic_ssim:.4} / {y_pred_ssim:.4},",
            )

            y_pred_psnr_metric.reset()
            y_pred_ssim_metric.reset()

            y_bicubic_psnr_metric.reset()
            y_bicubic_ssim_metric.reset()

            model.train()

        if epoch % args.checkpoint_interval == 0:
            checkpoint = {
                "epoch": epoch,
                "model_args": model_args,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            torch.save(checkpoint, args.checkpoint_path)

            print("Checkpoint saved")


if __name__ == "__main__":
    main()
