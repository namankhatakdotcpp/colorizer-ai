"""
GAN Trainer with R1 Gradient Penalty

Extends the production GANTrainer with R1 regularization for improved
discriminator stability and training convergence.

The R1 penalty is applied periodically (e.g., every 16 steps) rather than
every step to maintain computational efficiency.
"""

import logging
from typing import Dict, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from training.train_gan_production import GANTrainer, ExponentialMovingAverage
from training.r1_penalty import R1PenaltyCalculator

logger = logging.getLogger(__name__)


class GANTrainerWithR1(GANTrainer):
    """
    GAN Trainer with R1 gradient penalty support.
    
    Extends GANTrainer to add R1 regularization for discriminator.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        device: torch.device for training
        use_amp: Use mixed precision (default: True)
        use_ema: Use exponential moving average (default: True)
        ema_decay: EMA decay coefficient (default: 0.999)
        n_critic: Discriminator updates per generator update (default: 2)
        gradient_clip: Gradient clipping norm (default: 0.5)
        use_r1_penalty: Enable R1 penalty (default: True)
        lambda_r1: R1 penalty weight (default: 10.0)
        apply_r1_every_n_steps: Apply R1 every N discriminator steps (default: 16)
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        n_critic: int = 2,
        gradient_clip: float = 0.5,
        use_r1_penalty: bool = True,
        lambda_r1: float = 10.0,
        apply_r1_every_n_steps: int = 16,
    ):
        """Initialize GAN trainer with R1 penalty support."""
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            device=device,
            use_amp=use_amp,
            use_ema=use_ema,
            ema_decay=ema_decay,
            n_critic=n_critic,
            gradient_clip=gradient_clip,
        )
        
        # R1 penalty configuration
        self.use_r1_penalty = use_r1_penalty
        self.lambda_r1 = lambda_r1
        self.apply_r1_every_n_steps = apply_r1_every_n_steps
        self.step_count = 0
        
        # Initialize R1 calculator
        if self.use_r1_penalty:
            self.r1_calculator = R1PenaltyCalculator(
                self.discriminator,
                self.device,
                lambda_r1=self.lambda_r1
            )
            logger.info(
                f"✅ Initialized R1 penalty (lambda={self.lambda_r1}, "
                f"every {self.apply_r1_every_n_steps} steps)"
            )
        else:
            self.r1_calculator = None
    
    def _should_apply_r1(self) -> bool:
        """Check if R1 penalty should be applied this step."""
        if not self.use_r1_penalty:
            return False
        return self.step_count % self.apply_r1_every_n_steps == 0
    
    def train_step(
        self,
        condition: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step with R1 penalty support.
        
        Args:
            condition: Condition images (L channel)
            target: Target images (RGB)
        
        Returns:
            Loss dictionary with R1 penalty info if applied
        """
        batch_size = target.size(0)

        # Move to device
        condition = condition.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        # Label smoothing
        real_labels = torch.full((batch_size, 1), 0.9, device=self.device)
        fake_labels = torch.full((batch_size, 1), 0.1, device=self.device)

        loss_dict = {}

        # ===== DISCRIMINATOR STEPS (n_critic times) =====
        for _ in range(self.n_critic):
            self.optimizer_d.zero_grad(set_to_none=True)

            # Generate fake images (detach to avoid D->G gradients)
            with torch.no_grad():
                if self.use_amp:
                    with autocast():
                        fake_rgb = self.generator(condition)
                else:
                    fake_rgb = self.generator(condition)

            # Prepare inputs
            fake_input = torch.cat([condition, fake_rgb], dim=1)
            real_input = torch.cat([condition, target], dim=1)

            # Discriminator forward
            if self.use_amp:
                with autocast():
                    disc_fake_logits, disc_fake_features = self.discriminator(fake_input)
                    disc_real_logits, disc_real_features = self.discriminator(real_input)

                    # D loss
                    loss_d, dict_d = self.loss_manager(
                        disc_fake_logits=disc_fake_logits,
                        disc_real_logits=disc_real_logits,
                        real_labels=real_labels.expand_as(disc_real_logits[0]),
                        fake_labels=fake_labels.expand_as(disc_fake_logits[0]),
                        mode="discriminator",
                    )

                # Add R1 penalty if applicable
                if self._should_apply_r1():
                    r1_penalty, r1_stats = self.r1_calculator.compute_penalty_scaled(
                        real_input, self.scaler_d
                    )
                    loss_d = loss_d + r1_penalty
                    loss_dict.update(r1_stats)

                # Backward
                self.scaler_d.scale(loss_d).backward()
                self.scaler_d.unscale_(self.optimizer_d)

            else:
                disc_fake_logits, disc_fake_features = self.discriminator(fake_input)
                disc_real_logits, disc_real_features = self.discriminator(real_input)

                # D loss
                loss_d, dict_d = self.loss_manager(
                    disc_fake_logits=disc_fake_logits,
                    disc_real_logits=disc_real_logits,
                    real_labels=real_labels.expand_as(disc_real_logits[0]),
                    fake_labels=fake_labels.expand_as(disc_fake_logits[0]),
                    mode="discriminator",
                )

                # Add R1 penalty if applicable
                if self._should_apply_r1():
                    r1_penalty, r1_stats = self.r1_calculator.compute_penalty(real_input)
                    loss_d = loss_d + r1_penalty
                    loss_dict.update(r1_stats)

                loss_d.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                self.gradient_clip
            )

            # Optimizer step
            if self.use_amp:
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
            else:
                self.optimizer_d.step()

            loss_dict.update({f"d_{k}": v for k, v in dict_d.items()})

        # Increment step counter
        self.step_count += 1

        # ===== GENERATOR STEP =====
        self.optimizer_g.zero_grad(set_to_none=True)

        # Generate fake images
        if self.use_amp:
            with autocast():
                fake_rgb = self.generator(condition)
        else:
            fake_rgb = self.generator(condition)

        fake_input = torch.cat([condition, fake_rgb], dim=1)
        real_input = torch.cat([condition, target], dim=1)

        # Discriminator forward for G loss (no gradient into discriminator)
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    disc_real_logits, disc_real_features = self.discriminator(real_input)
            else:
                disc_real_logits, disc_real_features = self.discriminator(real_input)

        # Generator forward
        if self.use_amp:
            with autocast():
                disc_fake_logits, disc_fake_features = self.discriminator(fake_input)

                # G loss
                loss_g, dict_g = self.loss_manager(
                    generated=fake_rgb,
                    target=target,
                    disc_fake_logits=disc_fake_logits,
                    disc_real_logits=disc_real_logits,
                    fake_features=disc_fake_features,
                    real_features=disc_real_features,
                    mode="generator",
                )

            # Backward
            self.scaler_g.scale(loss_g).backward()
            self.scaler_g.unscale_(self.optimizer_g)

        else:
            disc_fake_logits, disc_fake_features = self.discriminator(fake_input)

            loss_g, dict_g = self.loss_manager(
                generated=fake_rgb,
                target=target,
                disc_fake_logits=disc_fake_logits,
                disc_real_logits=disc_real_logits,
                fake_features=disc_fake_features,
                real_features=disc_real_features,
                mode="generator",
            )

            loss_g.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.gradient_clip)

        # Optimizer step
        if self.use_amp:
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
        else:
            self.optimizer_g.step()

        # EMA update
        if self.use_ema:
            self.ema.update()

        loss_dict.update({f"g_{k}": v for k, v in dict_g.items()})

        return loss_dict


# ============================================================================
# Integration Utilities
# ============================================================================

def create_trainer_with_r1(
    generator: nn.Module,
    discriminator: nn.Module,
    device: Optional[torch.device] = None,
    use_amp: bool = True,
    use_ema: bool = True,
    use_r1: bool = True,
    lambda_r1: float = 10.0,
    r1_frequency: int = 16,
) -> GANTrainerWithR1:
    """
    Factory function to create GANTrainerWithR1.
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        device: torch.device for training
        use_amp: Mixed precision (default: True)
        use_ema: Exponential moving average (default: True)
        use_r1: R1 penalty (default: True)
        lambda_r1: R1 penalty weight (default: 10.0)
        r1_frequency: Apply R1 every N steps (default: 16)
    
    Returns:
        GANTrainerWithR1 instance ready for training
    """
    return GANTrainerWithR1(
        generator=generator,
        discriminator=discriminator,
        device=device,
        use_amp=use_amp,
        use_ema=use_ema,
        use_r1_penalty=use_r1,
        lambda_r1=lambda_r1,
        apply_r1_every_n_steps=r1_frequency,
    )
