import os
import shutil
from typing import Optional, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils


def _unwrap_state_dict(model: torch.nn.Module) -> dict:
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


class ModelTracker:
    """
    Tracks metrics and checkpoints for single-node and DDP training.
    """

    def __init__(self, run_name: str, checkpoint_dir: str = "checkpoints", log_dir: str = "tensorboard_logs"):
        self.run_name = run_name
        self.checkpoint_dir = checkpoint_dir
        self.save_path = os.path.join(checkpoint_dir, f"{run_name}_latest.pth")
        self.best_save_path = os.path.join(checkpoint_dir, f"{run_name}_best.pth")

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        best_metric: float,
        is_best: bool = False,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        extra_state: Optional[dict] = None,
    ) -> None:
        """
        Save checkpoint using recommended state_dict pattern.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": _unwrap_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if extra_state:
            checkpoint.update(extra_state)

        torch.save(checkpoint, self.save_path)

        if is_best:
            shutil.copyfile(self.save_path, self.best_save_path)
            print(f" -> Checkpoint [Epoch {epoch}] new best metric: {best_metric:.6f}")

    def attempt_resume(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        load_best: bool = False,
        device: str = "cuda",
    ) -> Tuple[int, float]:
        """
        Resume from latest or best checkpoint.
        Returns (start_epoch, best_metric).
        """
        target_path = self.best_save_path if load_best else self.save_path

        if not os.path.isfile(target_path):
            print(f"No checkpoint found at '{target_path}'. Starting from scratch.")
            return 0, float("inf")

        print(f"Resuming training from '{target_path}'")
        checkpoint = torch.load(target_path, map_location=device)

        model_state = checkpoint.get("model_state_dict", checkpoint)
        if hasattr(model, "module"):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0))
        best_metric = float(checkpoint.get("best_metric", float("inf")))
        return start_epoch, best_metric

    def log_metrics(self, epoch: int, train_loss: float, val_loss: float, psnr: float, ssim: float, lpips_score=None) -> None:
        self.writer.add_scalars("Loss/Epoch", {"Train": train_loss, "Validation": val_loss}, epoch)
        self.writer.add_scalar("Metrics/PSNR", psnr, epoch)
        self.writer.add_scalar("Metrics/SSIM", ssim, epoch)
        if lpips_score is not None:
            self.writer.add_scalar("Metrics/LPIPS", lpips_score, epoch)

    def log_images(self, epoch: int, real_lr: torch.Tensor, real_hr: torch.Tensor, synth_hr: torch.Tensor) -> None:
        n_images = min(4, synth_hr.size(0))

        input_grid = vutils.make_grid(real_lr[:n_images], nrow=n_images, normalize=True, scale_each=True)
        target_grid = vutils.make_grid(real_hr[:n_images], nrow=n_images, normalize=True, scale_each=True)
        pred_grid = vutils.make_grid(synth_hr[:n_images], nrow=n_images, normalize=True, scale_each=True)

        self.writer.add_image("Images_Input/Low_Resolution", input_grid, epoch)
        self.writer.add_image("Images_Target/High_Resolution", target_grid, epoch)
        self.writer.add_image("Images_Prediction/Generator_Output", pred_grid, epoch)

    def close(self) -> None:
        self.writer.close()
