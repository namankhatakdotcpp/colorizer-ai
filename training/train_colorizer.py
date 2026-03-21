import argparse
import contextlib
import hashlib
import math
import os
import shutil
import socket
import traceback
# Cap CPU thread fan-out to reduce contention on shared nodes.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Keep NCCL logs actionable and fail fast on async communicator errors.
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_TIMEOUT"] = "1800"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
import time
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import VGG16_Weights, vgg16

from datasets.dataset_colorizer import build_combined_colorization_dataset
from models.patch_discriminator import PatchDiscriminator
from models.unet_colorizer import UNetColorizer


STAGE_NAME = "stage1_colorizer"
DEFAULT_DATA_ROOTS = ("datasets/flickr2k", "datasets/coco", "datasets/div2k")
MIN_BATCHES_PER_EPOCH = 10
MIN_CHECKPOINT_MB = 80.0
MAX_CHECKPOINT_MB = 800.0
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
TENSOR_EXTENSIONS = {".pt", ".pth", ".npy", ".npz"}


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enabled = True
        try:
            backbone = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            # relu3_3 captures richer color/semantic regions than shallow relu2_2.
            self.features = backbone.features[:23].eval()
            for param in self.features.parameters():
                param.requires_grad = False
        except Exception as exc:
            # Keep training stable in offline environments where VGG weights are unavailable.
            print(f"[WARN] Failed to load pretrained VGG16 for perceptual loss: {exc}")
            self.enabled = False
            self.features = torch.nn.Identity()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    @staticmethod
    def _lab_to_rgb(l_channel: torch.Tensor, ab_channel: torch.Tensor) -> torch.Tensor:
        """
        Differentiable LAB->RGB conversion.
        l_channel: [0, 1], ab_channel: [-1, 1]
        """
        l_real = torch.clamp(l_channel, 0.0, 1.0) * 100.0
        ab_real = torch.clamp(ab_channel, -1.0, 1.0) * 128.0

        a = ab_real[:, 0:1, :, :]
        b = ab_real[:, 1:2, :, :]

        y = (l_real + 16.0) / 116.0
        x = (a / 500.0) + y
        z = y - (b / 200.0)
        xyz = torch.cat([x, y, z], dim=1)

        mask = xyz > 0.2068966
        xyz_cubic = torch.pow(torch.clamp(xyz, min=1e-6), 3.0)
        xyz_linear = (xyz - 16.0 / 116.0) / 7.787
        xyz = torch.where(mask, xyz_cubic, xyz_linear)

        xyz_x = xyz[:, 0:1, :, :] * 0.95047
        xyz_y = xyz[:, 1:2, :, :] * 1.00000
        xyz_z = xyz[:, 2:3, :, :] * 1.08883

        r = xyz_x * 3.2406 + xyz_y * -1.5372 + xyz_z * -0.4986
        g = xyz_x * -0.9689 + xyz_y * 1.8758 + xyz_z * 0.0415
        b_rgb = xyz_x * 0.0557 + xyz_y * -0.2040 + xyz_z * 1.0570
        rgb = torch.cat([r, g, b_rgb], dim=1)

        mask_rgb = rgb > 0.0031308
        rgb_gamma = 1.055 * torch.pow(torch.clamp(rgb, min=1e-8), 1.0 / 2.4) - 0.055
        rgb_linear = rgb * 12.92
        rgb = torch.where(mask_rgb, rgb_gamma, rgb_linear)
        return torch.clamp(rgb, 0.0, 1.0)

    def forward(self, pred_ab: torch.Tensor, target_ab: torch.Tensor, l_channel: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return pred_ab.new_tensor(0.0)

        pred_rgb = self._lab_to_rgb(l_channel, pred_ab)
        target_rgb = self._lab_to_rgb(l_channel, target_ab)

        pred_feat = self.features(self._normalize(pred_rgb))
        with torch.no_grad():
            target_feat = self.features(self._normalize(target_rgb))
        return F.l1_loss(pred_feat, target_feat)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: UNet colorizer DDP training")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--data-root", type=str, default=None, help="Legacy single-dataset override.")
    parser.add_argument("--data-roots", type=str, nargs="+", default=list(DEFAULT_DATA_ROOTS))
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--run-name", type=str, default=STAGE_NAME)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-log-interval", type=int, default=100)
    parser.add_argument("--perf-log-interval", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--perceptual-weight", type=float, default=0.5)
    parser.add_argument("--colorfulness-weight", type=float, default=1.0)
    parser.add_argument("--chroma-target-ratio", type=float, default=0.85)
    parser.add_argument("--gan-start-epoch", type=int, default=30)
    parser.add_argument("--gan-weight", type=float, default=0.1)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ddp-bucket-cap-mb", type=int, default=75)
    parser.add_argument("--disable-perceptual-loss", action="store_true")
    parser.add_argument("--resume", type=str, default="auto", help="Resume from a specific checkpoint path or 'auto'.")
    parser.add_argument("--finetune-epochs", type=int, default=0, help="If >0 and resuming, run only this many extra epochs.")
    parser.add_argument("--auto-optimize", action="store_true", help="Enable performance-oriented runtime hints.")
    parser.add_argument("--debug-grad", action="store_true", help="Enable periodic grad norm logging.")
    parser.add_argument("--compile", action="store_true", help="Compile model with torch.compile.")
    parser.add_argument("--dataset-cache", action="store_true", help="Enable dataset caching hook when available.")
    parser.add_argument("--cuda-graphs", action="store_true", help="Enable CUDA Graphs when static-shape safe.")
    parser.add_argument("--debug-single", action="store_true", help="Run single-GPU debug mode without DDP.")
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    loss_value: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "loss": float(loss_value),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, device: torch.device):
    checkpoint = torch.load(path, map_location=device)

    # ✅ Load model safely
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    # ✅ Load optimizer safely
    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as exc:
            print(f"[WARN] Skipping optimizer load: {exc}")
    else:
        print("[WARN] No optimizer state found. Using fresh optimizer.")

    # ✅ Safe epoch handling
    if "epoch" in checkpoint:
        start_epoch = int(checkpoint["epoch"]) + 1
    else:
        print("[WARN] No epoch found. Starting from epoch 0.")
        start_epoch = 0

    # ✅ Safe metric
    best_metric = float(checkpoint.get("best_metric", float("inf")))

    print(f"[INFO] Loaded checkpoint from {path}")
    print(f"[INFO] Starting from epoch {start_epoch}")

    return start_epoch, best_metric
# === RESUME SYSTEM START ===
def _strip_module_prefix_if_needed(state_dict: dict, model: torch.nn.Module) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    model_keys = list(model.state_dict().keys())
    if len(model_keys) == 0:
        return state_dict

    model_has_module_prefix = any(key.startswith("module.") for key in model_keys)
    checkpoint_has_module_prefix = any(key.startswith("module.") for key in state_dict.keys())
    if checkpoint_has_module_prefix and not model_has_module_prefix:
        return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
    return state_dict


def _warn(rank: int, message: str) -> None:
    if rank == 0:
        print(f"[WARN] {message}")


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device, non_blocking=True)


def build_optimizer(args: argparse.Namespace, params):
    if args.optimizer == "adam":
        return optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    return optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))


def load_checkpoint_for_resume(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rank: int,
):
    start_epoch = 0
    best_metric = float("inf")

    if not path.exists():
        _warn(rank, f"Checkpoint not found: {path}. Starting from epoch 0.")
        return start_epoch, best_metric

    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        _warn(rank, f"Failed to load checkpoint {path}: {exc}. Starting from epoch 0.")
        return start_epoch, best_metric

    model_state_dict = checkpoint.get("model_state_dict")
    if model_state_dict is None:
        _warn(rank, "Missing key 'model_state_dict' in checkpoint. Using randomly initialized model.")
    else:
        try:
            normalized_state = _strip_module_prefix_if_needed(model_state_dict, model)
            load_result = model.load_state_dict(normalized_state, strict=False)
            if len(load_result.missing_keys) > 0:
                _warn(rank, f"Missing model keys during resume: {load_result.missing_keys}")
            if len(load_result.unexpected_keys) > 0:
                _warn(rank, f"Unexpected model keys during resume: {load_result.unexpected_keys}")
        except Exception as exc:
            _warn(rank, f"Failed to load model_state_dict: {exc}. Continuing without model weights.")

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is None:
        _warn(rank, "Missing key 'optimizer_state_dict' in checkpoint. Optimizer will start fresh.")
    else:
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception as exc:
            _warn(rank, f"Failed to load optimizer_state_dict: {exc}. Optimizer will start fresh.")

    epoch_value = checkpoint.get("epoch", 0)
    try:
        start_epoch = int(epoch_value) + 1
    except Exception:
        _warn(rank, f"Invalid epoch value '{epoch_value}' in checkpoint. Starting from epoch 0.")
        start_epoch = 0

    try:
        best_metric = float(checkpoint.get("best_metric", float("inf")))
    except Exception:
        best_metric = float("inf")

    return start_epoch, best_metric


# === RESUME SYSTEM END ===


def validate_checkpoint_size(path: Path, min_mb: float = MIN_CHECKPOINT_MB, max_mb: float = MAX_CHECKPOINT_MB) -> float:
    checkpoint_size = path.stat().st_size / (1024 * 1024)

    if checkpoint_size < min_mb:
        raise RuntimeError(
            f"Checkpoint too small ({checkpoint_size:.2f} MB) at {path}. Model not saved correctly."
        )

    # ✅ DO NOT CRASH — just warn
    if checkpoint_size > max_mb:
        print(
            f"[WARN] Large checkpoint: {checkpoint_size:.2f} MB at {path} "
            f"(expected {min_mb:.0f}-{max_mb:.0f} MB, but allowed)"
        )

    return checkpoint_size


def compute_grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    count = 0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().abs().mean().item())
            count += 1
    return total if count > 0 else 0.0


def _log_exception(rank: int, context: str) -> None:
    print(f"[ERROR][rank={rank}] {context}\n{traceback.format_exc()}", flush=True)


def _validate_batch(batch, rank: int, context: str):
    if batch is None:
        raise RuntimeError(f"{context}: batch is None")
    if not isinstance(batch, (list, tuple)) or len(batch) != 2:
        raise RuntimeError(f"{context}: expected (l_channel, ab_target), got {type(batch)} with len={len(batch) if hasattr(batch, '__len__') else 'N/A'}")
    l_channel, ab_target = batch
    if l_channel is None or ab_target is None:
        raise RuntimeError(f"{context}: encountered None tensor in batch")
    if not torch.is_tensor(l_channel) or not torch.is_tensor(ab_target):
        raise RuntimeError(f"{context}: expected tensors, got {type(l_channel)} and {type(ab_target)}")
    if l_channel.ndim != 4 or ab_target.ndim != 4:
        raise RuntimeError(f"{context}: expected 4D tensors, got shapes {tuple(l_channel.shape)} and {tuple(ab_target.shape)}")
    if l_channel.shape[0] != ab_target.shape[0]:
        raise RuntimeError(f"{context}: batch size mismatch {l_channel.shape[0]} != {ab_target.shape[0]}")
    if l_channel.shape[1] != 1:
        raise RuntimeError(f"{context}: expected L channel with C=1, got {l_channel.shape[1]}")
    if ab_target.shape[1] != 2:
        raise RuntimeError(f"{context}: expected AB target with C=2, got {ab_target.shape[1]}")
    return l_channel, ab_target


def _validate_cache_roots(cache_roots: List[str], include_tensors: bool) -> None:
    for cache_root_str in cache_roots:
        cache_root = Path(cache_root_str)
        if not cache_root.exists():
            raise FileNotFoundError(f"Cache shard missing: {cache_root}")
        file_count = sum(1 for _ in _iter_cache_candidates(cache_root, include_tensors=include_tensors))
        if file_count == 0:
            raise RuntimeError(f"Cache shard is empty: {cache_root}")


def run_sanity_check(sample_l: torch.Tensor, model: torch.nn.Module, rank: int) -> None:
    with torch.no_grad():
        output = model(sample_l)
    if rank == 0:
        print(f"Sanity check output shape: {tuple(output.shape)}")


def make_worker_init_fn(rank: int, world_size: int, num_workers: int):
    def _init_fn(worker_id: int) -> None:
        if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
            return
        try:
            available_cores = sorted(list(os.sched_getaffinity(0)))
            if len(available_cores) == 0:
                return
            global_worker_id = (rank % max(world_size, 1)) * max(num_workers, 1) + worker_id
            target_core = available_cores[global_worker_id % len(available_cores)]
            os.sched_setaffinity(0, {target_core})
        except Exception:
            return

    return _init_fn


class PrefetchLoader:
    def __init__(self, loader: DataLoader, device: torch.device):
        if not getattr(loader, "pin_memory", False):
            raise RuntimeError("PrefetchLoader requires DataLoader(pin_memory=True).")
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device, priority=-1)
        self.next_l = None
        self.next_ab = None

    def __len__(self) -> int:
        return len(self.loader)

    def _preload(self, iterator) -> None:
        try:
            next_batch = next(iterator)
        except StopIteration:
            self.next_l = None
            self.next_ab = None
            return
        if not isinstance(next_batch, (list, tuple)) or len(next_batch) != 2:
            raise RuntimeError(f"PrefetchLoader expected (l_channel, ab_target), got {type(next_batch)}")
        next_l, next_ab = next_batch
        if next_l is None or next_ab is None:
            raise RuntimeError("PrefetchLoader got None tensors from upstream loader")

        with torch.cuda.stream(self.stream):
            self.next_l = next_l.to(self.device, non_blocking=True)
            self.next_ab = next_ab.to(self.device, non_blocking=True)

    def __iter__(self):
        iterator = iter(self.loader)
        self._preload(iterator)
        while self.next_l is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            l_channel = self.next_l
            ab_target = self.next_ab
            l_channel.record_stream(torch.cuda.current_stream(self.device))
            ab_target.record_stream(torch.cuda.current_stream(self.device))
            self._preload(iterator)
            yield l_channel, ab_target


def _iter_cache_candidates(root: Path, include_tensors: bool) -> Iterable[Path]:
    allowed_ext = IMAGE_EXTENSIONS | (TENSOR_EXTENSIONS if include_tensors else set())
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed_ext:
            yield path


def _rank_shard_for_path(path: Path, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()
    return int(digest, 16) % num_shards


def prepare_node_local_cache(
    data_roots: List[str],
    local_rank: int,
    local_world_size: int,
    include_tensors: bool = False,
) -> List[str]:
    node_name = socket.gethostname().split(".")[0]
    cache_base = Path("/tmp") / node_name
    cache_base.mkdir(parents=True, exist_ok=True)

    localized_roots: List[str] = []
    for root_str in data_roots:
        src_root = Path(root_str).resolve()
        src_name = src_root.name
        shard_root = cache_base / src_name / f"shard_{local_rank}"
        shard_root.parent.mkdir(parents=True, exist_ok=True)
        ready_file = shard_root / ".cache_ready"
        lock_file = shard_root.parent / f".shard_{local_rank}.lock"
        if ready_file.exists():
            localized_roots.append(str(shard_root))
            continue

        lock_fd = None
        wait_start = time.time()
        while lock_fd is None:
            if ready_file.exists():
                localized_roots.append(str(shard_root))
                break
            try:
                lock_fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.time() - wait_start > 1800:
                    raise TimeoutError(f"Timeout waiting for cache lock: {lock_file}")
                time.sleep(2)
                continue

        if ready_file.exists():
            continue

        try:
            tmp_root = shard_root.with_name(shard_root.name + ".tmp")
            if tmp_root.exists():
                shutil.rmtree(tmp_root)
            tmp_root.mkdir(parents=True, exist_ok=True)

            for source_file in _iter_cache_candidates(src_root, include_tensors):
                rel_path = source_file.relative_to(src_root)
                if _rank_shard_for_path(rel_path, max(local_world_size, 1)) != local_rank:
                    continue
                target_path = tmp_root / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, target_path)

            if shard_root.exists():
                shutil.rmtree(shard_root)
            os.replace(tmp_root, shard_root)
            ready_file.touch()
            localized_roots.append(str(shard_root))
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            if lock_file.exists():
                lock_file.unlink()

    return localized_roots


@record
def main(args: argparse.Namespace | None = None) -> None:
    args = parse_args() if args is None else args

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    data_roots: List[str] = [args.data_root] if args.data_root else args.data_roots
    if len(data_roots) == 0:
        raise RuntimeError("No dataset roots configured for Stage1 training.")
    for root in data_roots:
        if not Path(root).exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")

    ddp_enabled = not args.debug_single
    if ddp_enabled:
        required_env = ("LOCAL_RANK", "RANK", "WORLD_SIZE")
        missing = [k for k in required_env if k not in os.environ]
        if missing:
            raise RuntimeError(
                f"Missing torchrun environment variables: {missing}. "
                "Launch with torchrun or use --debug-single."
            )
        local_rank = int(os.environ["LOCAL_RANK"])
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        pg_timeout = torch.timedelta(seconds=1800) if hasattr(torch, "timedelta") else timedelta(seconds=1800)
        dist.init_process_group(backend="nccl", timeout=pg_timeout)
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        local_rank = 0
        local_world_size = 1
        rank = 0
        world_size = 1
        print("[DEBUG] Running --debug-single mode (DDP disabled).")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    try:
        if args.dataset_cache:
            try:
                data_roots = prepare_node_local_cache(
                    data_roots=data_roots,
                    local_rank=local_rank,
                    local_world_size=local_world_size,
                    include_tensors=True,
                )
                if ddp_enabled:
                    dist.barrier()
                node_name = socket.gethostname().split(".")[0]
                merged_roots: List[str] = []
                for original_root in ([args.data_root] if args.data_root else args.data_roots):
                    src_name = Path(original_root).resolve().name
                    shard_parent = Path("/tmp") / node_name / src_name
                    for shard_idx in range(max(local_world_size, 1)):
                        shard_dir = shard_parent / f"shard_{shard_idx}"
                        if shard_dir.exists():
                            merged_roots.append(str(shard_dir))
                if len(merged_roots) > 0:
                    data_roots = merged_roots
                _validate_cache_roots(data_roots, include_tensors=True)
            except Exception:
                _log_exception(rank, "dataset cache preparation failed")
                raise

        try:
            dataset, dataset_stats = build_combined_colorization_dataset(
                data_roots=data_roots,
                augment=True,
                image_size=args.image_size,
                warn_min_total_samples=50000,
            )
        except Exception:
            _log_exception(rank, "dataset creation failed")
            raise
        if len(dataset) == 0:
            raise RuntimeError("Dataset is empty")
        if rank == 0:
            print(f"Dataset length: {len(dataset)}")

        sampler = DistributedSampler(dataset, shuffle=True) if ddp_enabled else None

        cpu_count = os.cpu_count() or 1
        dynamic_workers = max(1, cpu_count // max(world_size, 1))
        num_workers = dynamic_workers if args.num_workers <= 0 else max(1, min(args.num_workers, dynamic_workers))
        if args.auto_optimize and rank == 0:
            print(
                f"[AUTO-OPTIMIZE] cpu_count={cpu_count}, world_size={world_size}, "
                f"selected_num_workers={num_workers}"
            )

        loader_kwargs = {
            "dataset": dataset,
            "batch_size": args.batch_size,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "worker_init_fn": make_worker_init_fn(rank=rank, world_size=world_size, num_workers=num_workers),
            "drop_last": False,
        }
        if sampler is not None:
            loader_kwargs["sampler"] = sampler
        else:
            loader_kwargs["shuffle"] = True

        try:
            raw_loader = DataLoader(**loader_kwargs)
            loader = PrefetchLoader(raw_loader, device=device)
        except Exception:
            _log_exception(rank, "dataloader creation failed")
            raise

        try:
            steps_per_epoch = len(loader)
            if steps_per_epoch == 0:
                raise RuntimeError("No batches produced")
            if steps_per_epoch < MIN_BATCHES_PER_EPOCH:
                raise RuntimeError("Dataset too small")

            first_batch = next(iter(raw_loader))
            first_l, first_ab = _validate_batch(first_batch, rank, "first batch validation")
            first_batch_shape = tuple(first_l.shape)
            image_resolution = f"{first_batch_shape[-2]}x{first_batch_shape[-1]}"
        except Exception:
            _log_exception(rank, "first batch fetch/validation failed")
            raise

        if rank == 0:
            print("Stage1 datasets:")
            combined_color_var = 0.0
            combined_images = 0
            combined_w = 0.0
            combined_h = 0.0
            for root, info in dataset_stats.items():
                size = int(info.get("size", 0))
                color_var = float(info.get("color_variance", 0.0))
                avg_res = info.get("average_resolution", (0.0, 0.0))
                avg_w = float(avg_res[0]) if isinstance(avg_res, (list, tuple)) and len(avg_res) >= 2 else 0.0
                avg_h = float(avg_res[1]) if isinstance(avg_res, (list, tuple)) and len(avg_res) >= 2 else 0.0
                res_dist = info.get("resolution_distribution", {})
                rejected_corrupt = int(info.get("rejected_corrupt", 0))
                rejected_grayscale = int(info.get("rejected_grayscale", 0))

                combined_color_var += color_var * size
                combined_images += size
                combined_w += avg_w * size
                combined_h += avg_h * size

                print(f"  - {root}: {size} images")
                print(
                    f"    color_variance={color_var:.6f}, "
                    f"avg_resolution=({avg_w:.1f}, {avg_h:.1f}), "
                    f"rejected_corrupt={rejected_corrupt}, rejected_grayscale={rejected_grayscale}"
                )
                print(f"    resolution_distribution={res_dist}")
            print(f"Dataset size: {len(dataset)}")
            if combined_images > 0:
                print(f"Color variance: {combined_color_var / combined_images:.6f}")
                print(
                    "Average resolution: "
                    f"({combined_w / combined_images:.1f}, {combined_h / combined_images:.1f})"
                )
            print(f"Batch size: {args.batch_size}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"GPUs: {world_size}")
            print(f"Image resolution: {image_resolution}")
            print(f"First batch shape: {first_batch_shape}")

        # === RESUME SYSTEM START ===
        resume_path = None
        if args.resume and args.resume.strip() and args.resume.lower() != "none":
            if args.resume.lower() == "auto":
                auto_candidates = [
                    Path(args.checkpoint_dir) / f"{args.run_name}_latest.pth",
                    Path(args.checkpoint_dir) / f"{args.run_name}_best.pth",
                ]
                resume_path = next((p for p in auto_candidates if p.exists()), None)
            else:
                resume_path = Path(args.resume)
        # === RESUME SYSTEM END ===
        ddp_bucket_cap_mb = max(50, min(int(args.ddp_bucket_cap_mb), 100))
        compile_enabled = False
        if args.compile and hasattr(torch, "compile"):
            major_cc, _ = torch.cuda.get_device_capability(device)
            compile_enabled = major_cc >= 8
            if rank == 0 and not compile_enabled:
                print("[WARN] torch.compile requested but GPU is pre-Ampere. Skipping compile.")
        if args.cuda_graphs and rank == 0:
            print("[WARN] CUDA Graphs require static-shape batches; currently disabled with drop_last=False.")

        if resume_path is not None:
            model = UNetColorizer(in_channels=1, out_channels=2)
            optimizer = build_optimizer(args, model.parameters())
            start_epoch, best_loss = load_checkpoint_for_resume(resume_path, model, optimizer, rank)
            if rank == 0:
                print(f"Loaded checkpoint: {resume_path}")
                print(f"Resuming training from epoch {start_epoch}")
            if ddp_enabled:
                dist.barrier()
            model = model.to(device)
            _move_optimizer_state_to_device(optimizer, device)
            if compile_enabled:
                model = torch.compile(model, mode="reduce-overhead")
            if ddp_enabled:
                model = DDP(
                    model,
                    device_ids=[local_rank],
                    broadcast_buffers=False,
                    gradient_as_bucket_view=True,
                    bucket_cap_mb=ddp_bucket_cap_mb,
                    find_unused_parameters=False,
                )
        else:
            model = UNetColorizer(in_channels=1, out_channels=2).to(device)
            if compile_enabled:
                model = torch.compile(model, mode="reduce-overhead")
            if ddp_enabled:
                model = DDP(
                    model,
                    device_ids=[local_rank],
                    broadcast_buffers=False,
                    gradient_as_bucket_view=True,
                    bucket_cap_mb=ddp_bucket_cap_mb,
                    find_unused_parameters=False,
                )
            optimizer = build_optimizer(args, model.parameters())

        total_params = sum(p.numel() for p in model.parameters())
        if total_params < 1_000_000:
            raise RuntimeError("Model architecture incorrect")
        if rank == 0:
            print(f"Model parameters: {total_params:,}")

        criterion = torch.nn.L1Loss().to(device)
        perceptual_loss_fn = VGGPerceptualLoss().to(device)
        perceptual_loss_fn.eval()
        if args.disable_perceptual_loss:
            perceptual_loss_fn.enabled = False
        if rank == 0 and not perceptual_loss_fn.enabled:
            print("[WARN] Perceptual loss disabled.")
        if rank == 0:
            print(f"Loss weights: L1=0.05, P={args.perceptual_weight}, C={args.colorfulness_weight}")

        # PatchGAN discriminator for adversarial color realism (activates after warmup epochs).
        disc = PatchDiscriminator(in_channels=3).to(device)
        if ddp_enabled:
            disc = DDP(
                disc,
                device_ids=[local_rank],
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                bucket_cap_mb=ddp_bucket_cap_mb,
                find_unused_parameters=False,
            )
        disc_optimizer = optim.AdamW(disc.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)
        gan_criterion = nn.BCEWithLogitsLoss().to(device)

        scaler = GradScaler(enabled=True)
        accum_steps = max(1, int(args.accum_steps))

        try:
            run_sanity_check(first_l.to(device, non_blocking=True), model, rank)
        except Exception:
            _log_exception(rank, "sanity forward pass failed")
            raise

        warmup_steps = min(3, steps_per_epoch)
        if warmup_steps > 0:
            try:
                model.eval()
                with torch.no_grad():
                    warmup_iter = iter(loader)
                    for _ in range(warmup_steps):
                        try:
                            warm_batch = next(warmup_iter)
                        except StopIteration:
                            break
                        warm_l, _ = _validate_batch(warm_batch, rank, "warmup batch validation")
                        with autocast(enabled=True):
                            _ = model(warm_l)
                torch.cuda.synchronize(device)
                model.train()
            except Exception:
                _log_exception(rank, "warmup forward failed")
                raise

        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        latest_path = ckpt_dir / f"{args.run_name}_latest.pth"
        best_path = ckpt_dir / f"{args.run_name}_best.pth"

        start_epoch = 0 if resume_path is None else start_epoch
        best_loss = float("inf") if resume_path is None else best_loss
        if resume_path is None and latest_path.exists():
            if rank == 0:
                print(f"Resuming from {latest_path}")
            start_epoch, best_loss = load_checkpoint(latest_path, model, optimizer, device)

        if args.finetune_epochs > 0 and start_epoch > 0:
            args.epochs = min(args.epochs, start_epoch + args.finetune_epochs)
            if rank == 0:
                print(f"[INFO] Fine-tune window enabled: training until epoch {args.epochs}.")

        warmup_epochs = 5
        total_epochs = max(int(args.epochs), 1)

        def lr_lambda(current_epoch: int) -> float:
            if current_epoch < warmup_epochs:
                return float(current_epoch + 1) / float(warmup_epochs)
            progress = (current_epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        if ddp_enabled:
            dist.barrier()

        if start_epoch >= args.epochs:
            raise RuntimeError(
                f"Resume epoch ({start_epoch}) >= target epochs ({args.epochs}). "
                "Increase --epochs or remove checkpoint to continue training."
            )

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            disc.train()
            torch.cuda.reset_peak_memory_stats(device)

            running_loss = 0.0
            running_l1_loss = 0.0
            running_perceptual_loss = 0.0
            running_colorfulness = 0.0
            running_gan_loss = 0.0
            running_disc_loss = 0.0
            running_data_time = 0.0
            running_compute_time = 0.0
            zero_grad_counter = 0
            grad_sync_steps = 0
            accum_counter = 0
            optimizer.zero_grad(set_to_none=True)
            step_clock = time.perf_counter()

            for step, batch in enumerate(loader, start=1):
                l_channel, ab_target = _validate_batch(batch, rank, f"train batch epoch={epoch + 1} step={step}")
                data_time = time.perf_counter() - step_clock
                running_data_time += data_time
                accum_counter += 1
                sync_gradients = accum_counter >= accum_steps or step == steps_per_epoch
                accumulation_divisor = accum_counter if sync_gradients else accum_steps
                compute_start = time.perf_counter()
                ddp_sync_context = contextlib.nullcontext() if (sync_gradients or not ddp_enabled) else model.no_sync()

                try:
                    with ddp_sync_context:
                        with autocast(enabled=True):
                            ab_pred = torch.clamp(model(l_channel), -1.0, 1.0)
                            l1_loss = criterion(ab_pred, ab_target)
                            p_loss = perceptual_loss_fn(ab_pred, ab_target, l_channel)

                            pred_chroma = torch.sqrt(torch.clamp(ab_pred[:, 0] ** 2 + ab_pred[:, 1] ** 2, min=1e-8))
                            target_chroma = torch.sqrt(torch.clamp(ab_target[:, 0] ** 2 + ab_target[:, 1] ** 2, min=1e-8))
                            chroma_ratio = pred_chroma / (target_chroma + 1e-6)
                            ratio_penalty = torch.relu(args.chroma_target_ratio - chroma_ratio).mean()
                            chroma_mse = F.mse_loss(pred_chroma, target_chroma)
                            pred_norm_a = ab_pred[:, 0] / (pred_chroma + 1e-6)
                            pred_norm_b = ab_pred[:, 1] / (pred_chroma + 1e-6)
                            tgt_norm_a = ab_target[:, 0] / (target_chroma + 1e-6)
                            tgt_norm_b = ab_target[:, 1] / (target_chroma + 1e-6)
                            hue_loss = (1.0 - (pred_norm_a * tgt_norm_a + pred_norm_b * tgt_norm_b)).clamp(min=0).mean()
                            colorfulness = ratio_penalty + 0.5 * chroma_mse + 0.3 * hue_loss

                            total_loss = (
                                0.05 * l1_loss + args.perceptual_weight * p_loss + args.colorfulness_weight * colorfulness
                            )

                            g_adv = l1_loss.new_tensor(0.0)
                            if epoch >= int(args.gan_start_epoch):
                                for param in disc.parameters():
                                    param.requires_grad_(False)
                                fake_rgb_gen = perceptual_loss_fn._lab_to_rgb(l_channel, ab_pred)
                                gen_logits = disc(fake_rgb_gen)
                                g_adv = gan_criterion(gen_logits, torch.ones_like(gen_logits))
                                for param in disc.parameters():
                                    param.requires_grad_(True)
                                total_loss = total_loss + float(args.gan_weight) * g_adv

                            loss = total_loss / accumulation_divisor

                    d_loss = l1_loss.new_tensor(0.0)
                    if epoch >= int(args.gan_start_epoch) and sync_gradients:
                        with torch.no_grad():
                            fake_rgb = perceptual_loss_fn._lab_to_rgb(l_channel, ab_pred.detach()).float()
                            real_rgb = perceptual_loss_fn._lab_to_rgb(l_channel, ab_target).float()

                        disc_optimizer.zero_grad(set_to_none=True)
                        real_logits = disc(real_rgb)
                        fake_logits = disc(fake_rgb)
                        d_loss = (
                            gan_criterion(real_logits, torch.ones_like(real_logits))
                            + gan_criterion(fake_logits, torch.zeros_like(fake_logits))
                        ) * 0.5
                        d_loss.backward()
                        torch.nn.utils.clip_grad_norm_(disc.parameters(), args.gradient_clip)
                        disc_optimizer.step()

                    if not torch.isfinite(total_loss):
                        raise RuntimeError("Non-finite loss detected")
                    scaler.scale(loss).backward()
                except Exception:
                    _log_exception(rank, f"forward/backward failed at epoch={epoch + 1}, step={step}")
                    raise

                compute_time = time.perf_counter() - compute_start
                running_compute_time += compute_time

                if sync_gradients:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

                    if args.debug_grad and step % max(1, args.grad_log_interval) == 0:
                        grad_norm = compute_grad_norm(model)
                        if rank == 0:
                            print(f"Grad norm (step {step}): {grad_norm:.6e}")
                        if grad_norm <= 1e-12:
                            zero_grad_counter += 1
                            if zero_grad_counter >= 5:
                                raise RuntimeError("Gradients are zero for 5 consecutive checks. Stopping training.")
                        else:
                            zero_grad_counter = 0

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0
                    grad_sync_steps += 1

                running_loss += float(total_loss.item())
                running_l1_loss += float(l1_loss.item())
                running_perceptual_loss += float(p_loss.item())
                running_colorfulness += float(colorfulness.item())
                running_gan_loss += float(g_adv.item())
                running_disc_loss += float(d_loss.item())
                if rank == 0 and step % max(1, args.perf_log_interval) == 0:
                    max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                    bottleneck = "data" if running_data_time > running_compute_time else "compute"
                    print(
                        f"[PERF] epoch={epoch + 1} step={step}/{steps_per_epoch} "
                        f"data={1000.0 * running_data_time / step:.1f}ms "
                        f"compute={1000.0 * running_compute_time / step:.1f}ms "
                        f"sync_steps={grad_sync_steps} "
                        f"max_mem={max_mem_gb:.2f}GB bottleneck={bottleneck}"
                    )
                step_clock = time.perf_counter()

            if ddp_enabled:
                stats = torch.tensor(
                    [
                        running_loss,
                        running_l1_loss,
                        running_perceptual_loss,
                        running_colorfulness,
                        running_gan_loss,
                        running_disc_loss,
                        running_data_time,
                        running_compute_time,
                    ],
                    device=device,
                )
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                denom = world_size * max(steps_per_epoch, 1)
                avg_loss = float(stats[0].item()) / denom
                avg_l1_loss = float(stats[1].item()) / denom
                avg_perceptual_loss = float(stats[2].item()) / denom
                avg_colorfulness = float(stats[3].item()) / denom
                avg_gan_loss = float(stats[4].item()) / denom
                avg_disc_loss = float(stats[5].item()) / denom
                avg_data_time = float(stats[6].item()) / denom
                avg_compute_time = float(stats[7].item()) / denom
            else:
                denom = max(steps_per_epoch, 1)
                avg_loss = running_loss / denom
                avg_l1_loss = running_l1_loss / denom
                avg_perceptual_loss = running_perceptual_loss / denom
                avg_colorfulness = running_colorfulness / denom
                avg_gan_loss = running_gan_loss / denom
                avg_disc_loss = running_disc_loss / denom
                avg_data_time = running_data_time / denom
                avg_compute_time = running_compute_time / denom

            epoch_time = time.time() - epoch_start

            if rank == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] | Loss: {avg_loss:.6f} "
                    f"(L1: {avg_l1_loss:.6f}, P: {avg_perceptual_loss:.6f}, Chroma: {avg_colorfulness:.6f}, "
                    f"GAdv: {avg_gan_loss:.6f}, D: {avg_disc_loss:.6f}) "
                    f"| Steps: {steps_per_epoch} | Time: {epoch_time:.1f}s | "
                    f"AvgData: {1000.0 * avg_data_time:.1f}ms | AvgCompute: {1000.0 * avg_compute_time:.1f}ms"
                )
                if epoch_time > 800.0:
                    suggested_workers = max(1, num_workers // 2) if num_workers > 0 else 0
                    print(
                        f"[WARN] Slow epoch detected ({epoch_time:.1f}s > 800s). "
                        f"Suggested --num-workers {suggested_workers} for next restart."
                    )

            if rank == 0:
                current_epoch = epoch
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss

                save_checkpoint(latest_path, current_epoch, model, optimizer, best_loss, avg_loss)
                size_mb = validate_checkpoint_size(latest_path, MIN_CHECKPOINT_MB)
                print(f"Checkpoint size: {size_mb:.2f} MB ({latest_path})")

                if is_best:
                    save_checkpoint(best_path, current_epoch, model, optimizer, best_loss, avg_loss)
                    validate_checkpoint_size(best_path, MIN_CHECKPOINT_MB)
            scheduler.step(epoch + 1)
            if rank == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"[LR] Epoch {epoch + 1}: lr={current_lr:.2e}")
            if ddp_enabled:
                dist.barrier()
    finally:
        if ddp_enabled and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    cli_args = parse_args()
    if not cli_args.debug_single and "WORLD_SIZE" not in os.environ:
        print(
            "Please run with torchrun, e.g. torchrun --nproc_per_node=4 training/train_colorizer.py "
            "or use --debug-single for single-GPU debugging."
        )
    else:
        main(cli_args)
