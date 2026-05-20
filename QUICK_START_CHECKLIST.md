# GAN Colorization Fix - Quick Start Checklist

**Action Items for Restarting Training with Fixes Applied**

---

## ✅ Pre-Training Verification (5 minutes)

### Verify All Fixes Applied:

- [ ] **Loss Manager Updated**
  ```bash
  python -c "from losses import create_loss_manager; lm = create_loss_manager(); print(f'lambda_l1={lm.lambda_l1}'); assert lm.lambda_l1 == 5.0"
  ```
  Should print: `lambda_l1=5.0`

- [ ] **Files Modified** 
  - [ ] `/losses/loss_manager.py` - lambda_l1 default changed
  - [ ] `/losses/examples.py` - examples updated  
  - [ ] `/debug_lab_scaling.py` - created for debugging

- [ ] **Debug Script Exists**
  ```bash
  python debug_lab_scaling.py
  ```
  Should run without errors and show LAB reconstruction tests

---

## 🚀 Training Configuration (Before Starting)

### Confirm Training Script Settings:

**File**: `/training/train_gan_refinement.py`

Verify these values in `__init__`:

```python
✓ lambda_l1: float = 5.0         # REDUCED from 50.0
✓ lambda_perceptual: float = 0.5 # Gentler perceptual
✓ learning_rate_g: float = 2e-5  # Generator
✓ learning_rate_d: float = 1e-5  # Discriminator (lower)
✓ n_critic: int = 1              # D updates per G update
✓ real_label_smooth = 0.85       # More uncertainty
✓ fake_label = 0.15              # More uncertainty
```

---

## 📊 Expected Epoch 0-5 Behavior

**IMPORTANT**: These are expected changes showing the fix is working:

### Losses (will change from previous runs):
- `loss_g_l1`: Will be **HIGHER** (less regularization) ✓ Expected
- `loss_g_adv`: Will be **HIGHER** (adversarial learning) ✓ Expected  
- `loss_g_perceptual`: May be **LOWER** (reduced weight) ✓ Expected
- `loss_d_real`: ~0.6-0.7 ✓ Good
- `loss_d_fake`: ~0.3-0.5 ✓ Good

### Visual Output:
- **Epoch 0-2**: Mostly grayscale with faint color tints
- **Epoch 3-5**: Visible color starting to appear
- **Epoch 5+**: Colors becoming more saturated

---

## 🔍 Monitoring During Training

### Check Every 5 Epochs:

1. **Visual Check** (most important):
   ```
   Look at sample outputs:
   ✓ Epoch 5:  Any visible colors yet? (may be faint)
   ✓ Epoch 10: Colors should be clearly visible now
   ✓ Epoch 20: Good color saturation
   ✓ Epoch 30+: Production quality
   ```

2. **Loss Check**:
   ```
   Verify loss_g_l1 is NOT dominant:
   - loss_g_l1 < 0.5 is GOOD
   - loss_g_l1 > 2.0 might need investigation
   ```

3. **AB Channel Check** (epoch 5, 10, 20):
   ```bash
   python debug_lab_scaling.py
   ```
   Should show AB channels with:
   - mean(|A|) > 0.1
   - mean(|B|) > 0.1
   - std(A) > 0.2
   - std(B) > 0.2

---

## ⚠️ If Grayscale Still Appears (Troubleshooting)

### Severity: 🔴 RED (Critical issue persisting)

**Step 1**: Verify loss manager change actually loaded
```python
# Check in training loop
print(f"Using lambda_l1 = {trainer.lambda_l1}")  # Should be 5.0 or lower
```

**Step 2**: Check generator output range
```python
# Add after generator forward pass
gen_output = generator(condition)
print(f"Gen output range: [{gen_output.min():.3f}, {gen_output.max():.3f}]")
print(f"Gen output std: {gen_output.std():.3f}")
# Should be roughly [-1, 1] with std > 0.3
```

**Step 3**: Increase color supervision
```python
# In trainer config:
lambda_l1 = 3.0              # Even lower L1
lambda_adversarial = 1.5     # Stronger adversarial
n_critic = 3                 # More discriminator updates
```

**Step 4**: Check if discriminator is working
```python
# Loss values to verify:
# D_real should be 0.6-0.7 (discriminator recognizes real)
# D_fake should be 0.3-0.5 (discriminator rejects fake initially)
# 
# If D_real > 0.8 and D_fake > 0.7:
#   → Discriminator is too weak or learning rate too high
#   → Try reducing lr_d or increasing n_critic
```

---

## ✨ Expected Improvement Over Epochs

| Epoch | Color Learning | Saturation | Quality |
|-------|---|---|---|
| 1 | 🔴 None | 0% | Grayscale |
| 5 | 🟡 Faint | ~10% | Some tints |
| 10 | 🟡 Starting | ~30% | Visible colors |
| 20 | 🟢 Good | ~60% | Realistic |
| 30+ | 🟢 Strong | ~80%+ | Production |

---

## 📋 Validation Checklist (Epoch 20)

At epoch 20, verify:

- [ ] Visual outputs show proper color saturation
- [ ] No pure grayscale outputs (should have rainbow colors)
- [ ] Colors match target distribution roughly
- [ ] Vertical stripe artifacts are minimal
- [ ] Loss values are stable (not exploding or collapsing)
- [ ] `loss_g_l1` is reasonable (<0.5)
- [ ] Debug script shows AB channels in good range

---

## 🛑 Critical Issues to Watch

### ❌ Issue: Loss goes to 0 instantly
- **Cause**: Generator learning degenerate solution
- **Fix**: Reduce learning rates (lr_g=1e-5, lr_d=5e-6)
- **Or**: Increase lambda_l1 back to 10.0

### ❌ Issue: Training crashes with NaN
- **Cause**: Numerical instability (rare)
- **Fix**: Reduce learning rates by 50%
- **Or**: Disable perceptual loss (lambda_perceptual=0)

### ❌ Issue: Discriminator loss decreases to 0.01
- **Cause**: Discriminator too strong
- **Fix**: Reduce n_critic (1 instead of 2)
- **Or**: Reduce learning rate ratio

### ✅ Issue: FFT warnings about NaN
- **Status**: NOT a critical issue
- **Impact**: Minimal (NaN protection is enabled)
- **Action**: Can ignore or disable FFT loss if annoying

---

## 🎯 Success Criteria

Training is working correctly when:

✅ After epoch 5:
- Generator predictions have visible color (not pure grayscale)
- Loss values are reasonable (<2.0 for each component)
- No NaN/Inf crashes

✅ After epoch 10:
- Clear colors visible in outputs
- Color saturation visible
- Loss curves smooth and stable

✅ After epoch 20:
- Properly saturated colors matching target
- Good perceptual quality
- Smooth convergence

---

## 🚀 Start Training

```bash
# Basic command
python -m training.train_gan_refinement \
    --data-dir data/colorized \
    --target-dir data/target \
    --num-epochs 120 \
    --batch-size 4 \
    --device cuda

# Or with specific settings
cd /usershome/cs671_user13/projects/colorizer-ai
source venv/bin/activate
python training/train_gan_refinement.py \
    --data-dir data/colorized_validation \
    --target-dir data/validation_targets \
    --num-epochs 120 \
    --batch-size 8 \
    --device cuda
```

---

## 📞 Support

If training doesn't improve after 20 epochs:

1. Verify loss manager has lambda_l1=5.0 (critical!)
2. Check that generator output uses Tanh (produces [-1,1])
3. Run debug script on epoch 10 samples
4. Review loss values - which loss is dominant?
5. Consider adjusting learning rates or increasing n_critic

**Remember**: The fix is reducing excessive L1 regularization that was forcing the generator toward grayscale. Training should show visible improvement within first 10-20 epochs.

---

*Last Updated: 20 May 2026*  
*Critical Priority: HIGH*  
*Estimated Resolution Time: 30-60 minutes training time + monitoring*
