import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import shutil

class ModelTracker:
    """
    Robust PyTorch model tracking system.
    Handles continuous metric logging to TensorBoard and safe Checkpoint resumption.
    """
    def __init__(self, run_name, checkpoint_dir="checkpoints", log_dir="tensorboard_logs"):
        self.run_name = run_name
        self.checkpoint_dir = checkpoint_dir
        self.save_path = os.path.join(checkpoint_dir, f"{run_name}_latest.pth")
        self.best_save_path = os.path.join(checkpoint_dir, f"{run_name}_best.pth")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize TensorBoard Writer
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))

    def save_checkpoint(self, epoch, model, optimizer, best_metric, is_best=False, scheduler=None):
        """
        Saves full dictionary state allowing flawless resumption.
        """
        # Handle DataParallel/DistributedDataParallel model wrappers cleanly
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        state = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': best_metric
        }
        
        if scheduler is not None:
             state['scheduler_state_dict'] = scheduler.state_dict()
             
        # Always save latest to prevent massive loss of progress on crashes
        torch.save(state, self.save_path)
        
        if is_best:
            # Atomic copy to guarantee integrity of the best model
            shutil.copyfile(self.save_path, self.best_save_path)
            print(f" -> Checkpoint [Epoch {epoch}]: New Best Metric '{best_metric:.4f}' recorded!")

    def attempt_resume(self, model, optimizer, scheduler=None, load_best=False, device='cuda'):
        """
        Restores model, optimizer, and training loop parameters.
        Returns: start_epoch, best_metric
        """
        target_path = self.best_save_path if load_best else self.save_path
        
        if not os.path.isfile(target_path):
            print(f"No checkpoint found at '{target_path}'. Starting from scratch.")
            return 0, 0.0 # Default starting logic
            
        print(f"Resuming training from '{target_path}'...")
        checkpoint = torch.load(target_path, map_location=device)
        
        # Load weights
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            
        # Load logic states
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint['epoch'] + 1, checkpoint.get('best_metric', 0.0)

    def log_metrics(self, epoch, train_loss, val_loss, psnr, ssim, lpips_score=None):
        """Standard floating point line graphs."""
        self.writer.add_scalars('Loss/Epoch', {'Train': train_loss, 'Validation': val_loss}, epoch)
        self.writer.add_scalar('Metrics/PSNR', psnr, epoch)
        self.writer.add_scalar('Metrics/SSIM', ssim, epoch)
        if lpips_score is not None:
            self.writer.add_scalar('Metrics/LPIPS', lpips_score, epoch)

    def log_images(self, epoch, real_lr, real_hr, synth_hr):
        """
        Visually logs a grid of input, ground truth, and predicted patches.
        Assumes tensors are bounded [0, 1] normalized.
        """
        # Ensure only up to 4 images max are displayed per log to save storage
        n_images = min(4, synth_hr.size(0))
        
        input_grid = vutils.make_grid(real_lr[:n_images], nrow=n_images, normalize=True, scale_each=True)
        target_grid = vutils.make_grid(real_hr[:n_images], nrow=n_images, normalize=True, scale_each=True)
        pred_grid = vutils.make_grid(synth_hr[:n_images], nrow=n_images, normalize=True, scale_each=True)
        
        self.writer.add_image('Images_Input/Low_Resolution', input_grid, epoch)
        self.writer.add_image('Images_Target/High_Resolution', target_grid, epoch)
        self.writer.add_image('Images_Prediction/Generator_Output', pred_grid, epoch)
        
    def close(self):
        """Flushes tensorboard buffer safely."""
        self.writer.close()
