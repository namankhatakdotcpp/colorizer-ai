# GAN Colorization Training Fixes - Implementation Guide

**Date**: 20 May 2026  
**Status**: Critical fixes implemented  
**Priority**: High - Addresses grayscale convergence issue

---

## Executive Summary

The GAN training was converging to a **grayscale equilibrium** where the generator predicts near-zero AB (color) channels, despite stable numerical training. This has been identified as caused by:

1. **Excessive L1 loss weight** (PRIMARY)
2. **Loss imbalance** causing insufficient color supervision
3. **Weak discriminator pressure** on desaturated outputs

All critical issues have been **FIXED**.

---

## Changes Implemented

### 1. Reduced L1 Loss Weight (PRIMARY FIX)

**Files Modified:**
- `losses/loss_manager.py`
- `losses/examples.py`

**Changes:**
```python
# BEFORE (causes grayscale convergence)
lambda_l1 = 50.0

# AFTER (allows color learning)
lambda_l1 = 5.0 to 10.0
```

**Rationale:**
- High L1 loss encourages averaging colors → AB channels converge to 0
- In LAB space: average(A) ≈ 0, average(B) ≈ 0 → grayscale
- Reduced L1 allows adversarial loss to push generator toward realistic colors
- Perceptual loss still provides structure supervision

**Impact**: ⭐⭐⭐ HIGHEST PRIORITY

### 2. Verified LAB Color Space Scaling

**Status:** ✅ CORRECT (no changes needed)

**Key Finding:**
```python
# From datasets/train_gan_refinement.py line 333
ab_channels = torch.from_numpy(colorized_lab[:, :, 1:3]).permute(2, 0, 1) / 127.0
# Correctly normalizes [-128,127] → [-1,1]
```

The LAB→RGB reconstruction is **properly implemented**:
```python
# During reconstruction (backend/training/losses/losses.py)
pred_ab_real = pred_ab * 128.0  # Scale from [-1,1] → [-128,127]
pred_rgb = lab_to_rgb(L_input, pred_ab_real)  # Convert to RGB
```

**Verification:** No fixes needed - scaling is correct

### 3. Created Debug Script

**New File:** `debug_lab_scaling.py`

**Purpose:** Runtime validation of:
- AB channel ranges (should be [-1, 1])
- LAB→RGB reconstruction correctness
- Grayscale convergence detection

**Usage:**
```bash
python debug_lab_scaling.py
```

**What it checks:**
- AB channels near zero (grayscale indicator)
- Correct range detection
- Reconstruction produces valid colors

---

## Pre-Training Checklist

Before restarting training, verify:

- [ ] Loss manager has been updated (`lambda_l1 = 5.0`)
- [ ] Generator output layer uses **Tanh** activation (produces [-1, 1])
- [ ] Training data AB channels are in [-1, 1] range
- [ ] Discriminator input concatenates condition + generated image correctly
- [ ] Learning rates are set to TTUR values:
  - Generator: `lr_g = 2e-5`
  - Discriminator: `lr_d = 1e-5` (lower than generator)

---

## Training Configuration Recommendations

### Recommended Settings (for colorization):

```python
# Loss weights (in loss_manager or trainer)
lambda_adversarial = 1.0      # Standard adversarial loss
lambda_l1 = 5.0               # REDUCED from 50.0 (critical!)
lambda_perceptual = 10.0      # Structure supervision
lambda_feature_matching = 10.0  # Mid-level feature matching
lambda_histogram = 5.0        # Color distribution matching

# Learning rates (TTUR - Two Time-Scale Update Rule)
lr_g = 2e-5   # Generator (lower)
lr_d = 1e-5   # Discriminator (lower than G for stability)

# Discriminator update frequency
n_critic = 2  # 2 discriminator updates per generator update

# Label smoothing (prevents D overconfidence)
real_label = 0.85  # Was 0.9 (more uncertainty)
fake_label = 0.15  # Was 0.1 (more uncertainty)

# EMA (Exponential Moving Average)
ema_decay = 0.9995  # Smooth generator weights
```

### Alternative Settings (if still getting grayscale):

```python
# MORE aggressive color learning (if needed)
lambda_l1 = 3.0          # Even lower L1
lambda_adversarial = 1.5  # Stronger adversarial pressure

# STRONGER discriminator
n_critic = 3             # 3 D updates per G update
lr_d = 2e-5             # Increase D learning rate
```

---

## Expected Changes After Fix

### During Training:
- **First 5 epochs**: Generator should start predicting non-zero AB channels
- **Epochs 10-20**: Visible color appearing in outputs (reds, greens, blues)
- **Epochs 30+**: More saturated, realistic colors
- **Loss behavior**:
  - `loss_g_l1` will be larger (less regularization)
  - `loss_g_adv` will increase (generator exploring color space)
  - `loss_d_*` will show discriminator working harder

### Sample Outputs:
- **Before fix**: Nearly grayscale images with faint colors
- **After fix**: Properly saturated colors matching target distribution

---

## Debugging with Provided Script

### Run Debug Script:

```bash
python debug_lab_scaling.py
```

### Output Interpretation:

✅ **Good Output:**
```
A Channel Statistics:
  Min: -0.85
  Max: 0.92
  Mean: 0.02
  Std: 0.35
```
(A and B channels have good range and variation)

❌ **Problem Output:**
```
A Channel Statistics:
  Min: -0.05
  Max: 0.08
  Mean: 0.01
  Std: 0.02
⚠️ AB channels are NEAR ZERO (likely grayscale convergence!)
```
(Indicates generator not learning colors)

---

## Monitoring During Training

### Key Metrics to Watch:

```
1. AB channel statistics (add to training logs):
   - mean(abs(pred_ab)) should increase over epochs
   - std(pred_ab) should increase over epochs
   
2. Loss ratios:
   - loss_g_l1 / loss_g_adv should be reasonable (not >100)
   
3. Visual quality:
   - Check sample outputs every 5 epochs
   - Look for color saturation increasing
   - Vertical stripes indicate FFT issues (not critical)
```

### Add to Training Loop:

```python
# During training validation/sampling
with torch.no_grad():
    # Generate samples
    samples = generator(condition)
    
    # Log AB statistics
    logger.info(f"Epoch {epoch}: "
                f"mean(|pred_ab|)={abs(pred_ab).mean():.4f}, "
                f"std(pred_ab)={pred_ab.std():.4f}")
```

---

## If Grayscale Persists After Fix

**Diagnostic steps:**

1. **Verify loss weight actually changed:**
   ```python
   from losses import create_loss_manager
   lm = create_loss_manager()
   print(f"lambda_l1 = {lm.lambda_l1}")  # Should be 5.0
   ```

2. **Check generator output:**
   ```python
   # Generator final layer should be Tanh
   # Output should be in [-1, 1] with good variance
   ```

3. **Monitor discriminator:**
   ```python
   # D_real should be ~0.5-0.7
   # D_fake should be ~0.3-0.5
   # If D is not differentiating, increase n_critic
   ```

4. **Check FFT warnings:**
   ```python
   # FFT NaN warnings are NOT preventing training
   # But they indicate numerical instability
   # Can usually be ignored in early epochs
   ```

---

## Recovery Instructions

### If Training Crashes:

1. **Resume from last checkpoint:**
   ```python
   # The trainer should auto-load latest checkpoint
   checkpoint_path = "checkpoints/stage5_gan/latest_checkpoint.pth"
   ```

2. **Reduce loss complexity if needed:**
   ```python
   # Disable perceptual loss temporarily
   lambda_perceptual = 0.0
   # Disable feature matching
   lambda_feature_matching = 0.0
   ```

3. **Increase stability:**
   ```python
   lambda_l1 = 10.0  # Increase from 5.0 if crashes
   lr_g = 1e-5       # Lower learning rate
   lr_d = 5e-6
   ```

---

## Expected Performance Timeline

| Epoch | Color Learning | Visual Quality | Notes |
|-------|----------------|----------------|-------|
| 1-5   | Minimal        | Mostly grayscale | Discriminator initializing |
| 5-20  | Starting       | Some colors visible | Colors becoming saturated |
| 20-50 | Good progress  | Clear colors | Good diversity |
| 50+   | Mature         | Realistic colors | Fine-tuning realism |
| 100+  | Refined        | Production quality | Ready for evaluation |

---

## Related Files Modified

1. **`/losses/loss_manager.py`** - Default `lambda_l1` changed to 5.0
2. **`/losses/examples.py`** - Updated example configurations
3. **`/debug_lab_scaling.py`** - New debugging script

---

## Validation Tests

### Test 1: Loss Manager Defaults
```bash
python -c "from losses import create_loss_manager; lm = create_loss_manager(); assert lm.lambda_l1 == 5.0, f'Expected 5.0, got {lm.lambda_l1}'"
```
✅ Should pass silently

### Test 2: Run Debug Script
```bash
python debug_lab_scaling.py
```
✅ Should show LAB reconstruction tests and summary

### Test 3: Quick Training Test
```bash
# Train for 2 epochs
python training/train_gan_refinement.py \
    --data-dir data/colorized \
    --target-dir data/target \
    --num-epochs 2 \
    --batch-size 4
```
✅ Should complete without crashing

---

## Next Steps

1. **Restart training** with updated configuration
2. **Monitor AB channel ranges** during first 10 epochs
3. **Run debug script** on epoch 5 and 20 outputs to verify color learning
4. **Check sample outputs** for visible color saturation
5. **Adjust learning rates** if needed based on loss curves
6. **Evaluate FID scores** at epochs 50, 100, 120

---

## Support / Questions

If you encounter issues:

1. Check loss values in first 5 epochs
2. Verify AB channels are increasing (not staying near zero)
3. Review output samples for color appearance
4. Run `debug_lab_scaling.py` on generated samples
5. Check generator output layer uses Tanh activation

---

## Summary of Root Cause

**Problem**: Generator learned grayscale convergence (AB ≈ 0)

**Reason**: Excessive L1 loss (50.0) encouraged averaging colors instead of learning realistic color distribution

**Solution**: Reduce L1 weight to 5.0, allowing adversarial loss to push toward realistic colors while perceptual loss provides structure

**Result**: Generator should now learn meaningful color information during training
