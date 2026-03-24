#!/usr/bin/env python3
"""
Quick test for semantic colorizer upgrade.

Tests:
1. Load baseline checkpoint
2. Convert to semantic model
3. Run inference with scene classification
4. Compare baseline vs. semantic
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

from models.unet_colorizer_semantic import UNetColorizerSemantic, UNetColorizer
from utils.colorizer_inference import ColorizerFactory, SemanticColorizerInference


def test_model_loading():
    """Test loading baseline and semantic models."""
    print("\n" + "="*60)
    print("TEST 1: Model Loading")
    print("="*60)
    
    device = torch.device("cpu")  # Use CPU for testing
    baseline_checkpoint = "checkpoints/stage1_colorizer_latest.pth"
    
    # Check if checkpoint exists
    if not Path(baseline_checkpoint).exists():
        print(f"❌ Baseline checkpoint not found: {baseline_checkpoint}")
        return False
    
    try:
        # Load baseline model
        print("\n[1] Loading baseline model...")
        baseline_model = UNetColorizer().to(device)
        baseline_model.eval()
        print("✅ Baseline model created")
        
        # Load semantic model from baseline checkpoint
        print("[2] Converting baseline to semantic model...")
        semantic_model = UNetColorizerSemantic.from_baseline_checkpoint(
            baseline_checkpoint,
            use_attention=True,
        ).to(device)
        print("✅ Semantic model created and weights loaded")
        
        # Check model sizes
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        semantic_params = sum(p.numel() for p in semantic_model.parameters())
        print(f"\n📊 Model Sizes:")
        print(f"   Baseline:  {baseline_params:,} parameters")
        print(f"   Semantic:  {semantic_params:,} parameters (+{semantic_params-baseline_params:,})")
        print(f"   Overhead:  {(semantic_params-baseline_params)/baseline_params*100:.1f}%")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test inference on dummy input."""
    print("\n" + "="*60)
    print("TEST 2: Inference on Dummy Input")
    print("="*60)
    
    device = torch.device("cpu")
    baseline_checkpoint = "checkpoints/stage1_colorizer_latest.pth"
    
    try:
        # Create dummy input
        print("\n[1] Creating dummy L channel (256x256)...")
        l_channel = torch.rand(1, 1, 256, 256) * 1.0  # L in [0, 1]
        print(f"✅ Input shape: {l_channel.shape}")
        
        # Baseline inference
        print("\n[2] Baseline inference...")
        baseline_model = UNetColorizer().to(device)
        baseline_model.eval()
        with torch.no_grad():
            ab_baseline = baseline_model(l_channel)
        print(f"✅ Output shape: {ab_baseline.shape}")
        print(f"   AB range: [{ab_baseline.min():.3f}, {ab_baseline.max():.3f}]")
        
        # Semantic inference
        print("\n[3] Semantic inference...")
        semantic_model = UNetColorizerSemantic.from_baseline_checkpoint(
            baseline_checkpoint,
            use_attention=True,
        ).to(device)
        semantic_model.eval()
        with torch.no_grad():
            ab_semantic, scene_logits = semantic_model(l_channel, return_semantic=True)
        print(f"✅ AB output shape: {ab_semantic.shape}")
        print(f"   AB range: [{ab_semantic.min():.3f}, {ab_semantic.max():.3f}]")
        print(f"✅ Scene logits shape: {scene_logits.shape}")
        
        # Scene classification
        scene_probs = torch.softmax(scene_logits, dim=1)[0]
        scene_classes = semantic_model.SCENE_CLASSES
        print(f"\n📊 Scene Classification:")
        for idx, cls in enumerate(scene_classes):
            print(f"   {cls:12s}: {scene_probs[idx]:.1%}")
        
        predicted_class = scene_classes[torch.argmax(scene_probs)]
        confidence = scene_probs[torch.argmax(scene_probs)].item()
        print(f"\n🎯 Predicted: {predicted_class} ({confidence:.1%} confidence)")
        
        # Compare differences
        ab_diff = (ab_semantic - ab_baseline).abs().mean()
        print(f"\n📈 AB Difference (semantic vs baseline): {ab_diff:.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factory():
    """Test ColorizerFactory auto-detection."""
    print("\n" + "="*60)
    print("TEST 3: ColorizerFactory Auto-Detection")
    print("="*60)
    
    device = torch.device("cpu")
    baseline_checkpoint = "checkpoints/stage1_colorizer_latest.pth"
    
    try:
        print("\n[1] Loading with mode='auto' on baseline checkpoint...")
        model = ColorizerFactory.load_model(baseline_checkpoint, device, mode="auto")
        is_semantic = isinstance(model, UNetColorizerSemantic)
        model_type = "semantic" if is_semantic else "baseline"
        print(f"✅ Detected: {model_type} model")
        
        # Test inference
        l_channel = torch.rand(1, 1, 256, 256)
        print("\n[2] Running inference...")
        if is_semantic:
            ab_pred, scene_info = ColorizerFactory.inference_semantic(model, l_channel, device)
            print(f"✅ Semantic inference successful")
            print(f"   Top scene: {scene_info['class_name']} ({scene_info['confidence']:.1%})")
        else:
            ab_pred = ColorizerFactory.inference_baseline(model, l_channel, device)
            print(f"✅ Baseline inference successful")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_high_level_api():
    """Test high-level SemanticColorizerInference API."""
    print("\n" + "="*60)
    print("TEST 4: High-Level API")
    print("="*60)
    
    baseline_checkpoint = "checkpoints/stage1_colorizer_latest.pth"
    
    try:
        print("\n[1] Initializing SemanticColorizerInference...")
        inferencer = SemanticColorizerInference(baseline_checkpoint)
        print(f"✅ Inferencer initialized (device: {inferencer.device})")
        print(f"   Model type: {'semantic' if inferencer.is_semantic else 'baseline'}")
        
        # Test colorization
        print("\n[2] Testing colorization...")
        l_channel = torch.rand(1, 1, 256, 256)
        ab_pred, scene_info = inferencer.colorize(l_channel)
        print(f"✅ Colorization successful")
        print(f"   AB shape: {ab_pred.shape}")
        
        if scene_info:
            print(f"\n📊 Scene Information:")
            print(f"   Class: {scene_info['class_name']}")
            print(f"   Confidence: {scene_info['confidence']:.1%}")
            print(f"   All probabilities:")
            for cls, prob in scene_info['all_probs'].items():
                print(f"      {cls:12s}: {prob:.2%}")
        
        # Test scene info only
        print("\n[3] Getting scene info only (faster)...")
        scene_info_only = inferencer.get_scene_info(l_channel.squeeze())
        if "error" not in scene_info_only:
            print(f"✅ Scene info retrieved")
        else:
            print(f"⚠️  {scene_info_only['error']}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SEMANTIC COLORIZER QUICK TEST")
    print("="*60)
    
    results = {
        "Model Loading": test_model_loading(),
        "Inference": test_inference(),
        "Factory": test_factory(),
        "High-Level API": test_high_level_api(),
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8s} {test_name}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nNext: Fine-tune the semantic model!")
        print("  python training/finetune_semantic_colorizer.py --epochs 25")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nCheck errors above and troubleshoot.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
