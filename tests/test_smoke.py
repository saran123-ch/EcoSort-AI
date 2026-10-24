"""
EcoSort AI - Smoke Tests
==========================
Lightweight tests to verify core functionality without requiring
the full dataset or a fully trained model.

Usage:
    python -m tests.test_smoke
"""

import os
import sys
import json
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_imports():
    """Test that config module loads without error."""
    from src.config import (
        CLASS_NAMES, NUM_CLASSES, IMAGE_SIZE, CONFIDENCE_THRESHOLD,
        SORTING_GUIDANCE, DATASET_CITATION
    )
    assert NUM_CLASSES == 9, f"Expected 9 classes, got {NUM_CLASSES}"
    assert len(CLASS_NAMES) == 9
    assert IMAGE_SIZE == 224
    assert 0 < CONFIDENCE_THRESHOLD < 1
    assert len(SORTING_GUIDANCE) == 9
    assert len(DATASET_CITATION) > 0
    print("  [PASS] Config imports and values correct")


def test_model_creation():
    """Test that the model can be created (without pretrained weights)."""
    from src.model import create_model
    model = create_model(pretrained=False)
    # Check output layer has 9 classes
    last_layer = model.classifier[-1]
    assert last_layer.out_features == 9, f"Expected 9 outputs, got {last_layer.out_features}"
    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable < total, "Some params should be frozen"
    print(f"  [PASS] Model created — {total:,} total params, {trainable:,} trainable")


def test_model_save_load():
    """Test model save and load roundtrip."""
    import torch
    from src.model import create_model, save_model, load_model

    model = create_model(pretrained=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model.pth")
        save_model(model, path)
        assert os.path.exists(path), "Model file not created"
        loaded = load_model(path)
        assert loaded is not None
    print("  [PASS] Model save/load roundtrip works")


def test_preprocessing():
    """Test image preprocessing produces correct tensor shape."""
    from PIL import Image
    from src.predict import preprocess_image

    # Create a dummy image
    img = Image.new("RGB", (512, 512), color=(128, 128, 128))
    tensor = preprocess_image(img)
    assert tensor.shape == (1, 3, 224, 224), f"Unexpected shape: {tensor.shape}"
    print("  [PASS] Image preprocessing produces correct shape (1, 3, 224, 224)")


def test_prediction_pipeline():
    """Test end-to-end prediction with a dummy model."""
    import torch
    from PIL import Image
    from src.model import create_model
    from src.predict import predict
    from src.config import CLASS_NAMES

    model = create_model(pretrained=False)
    model.eval()

    img = Image.new("RGB", (300, 300), color=(100, 150, 200))
    result = predict(model, img, device=torch.device("cpu"))

    assert "predicted_class" in result
    assert "confidence" in result
    assert "top3" in result
    assert "is_confident" in result
    assert "sorting_guidance" in result
    assert len(result["top3"]) == 3
    assert result["raw_predicted_class"] in CLASS_NAMES
    assert "decision_status" in result
    print(f"  [PASS] Prediction pipeline works — status: {result['decision_status']}, "
          f"raw class: {result['raw_predicted_class']}, reported: {result['predicted_class']} "
          f"({result['confidence']:.2%})")


def test_invalid_image_handling():
    """Test that non-RGB images are handled gracefully."""
    import torch
    from PIL import Image
    from src.model import create_model
    from src.predict import predict

    model = create_model(pretrained=False)
    model.eval()

    # Grayscale image
    gray_img = Image.new("L", (200, 200), color=128)
    result = predict(model, gray_img, device=torch.device("cpu"))
    assert "predicted_class" in result
    print("  [PASS] Grayscale image handled (auto-converted to RGB)")

    # RGBA image
    rgba_img = Image.new("RGBA", (200, 200), color=(128, 128, 128, 255))
    result = predict(model, rgba_img, device=torch.device("cpu"))
    assert "predicted_class" in result
    print("  [PASS] RGBA image handled (auto-converted to RGB)")


def test_transforms():
    """Test that train and eval transforms produce correct shapes."""
    from PIL import Image
    from src.dataset import get_train_transforms, get_eval_transforms

    img = Image.new("RGB", (500, 400))
    train_t = get_train_transforms()(img)
    eval_t = get_eval_transforms()(img)
    assert train_t.shape == (3, 224, 224), f"Train transform shape: {train_t.shape}"
    assert eval_t.shape == (3, 224, 224), f"Eval transform shape: {eval_t.shape}"
    print("  [PASS] Transforms produce correct output shape (3, 224, 224)")


def test_sorting_guidance_completeness():
    """Test that every class has sorting guidance."""
    from src.config import CLASS_NAMES, SORTING_GUIDANCE
    for cls in CLASS_NAMES:
        assert cls in SORTING_GUIDANCE, f"Missing guidance for {cls}"
        g = SORTING_GUIDANCE[cls]
        assert "bin" in g
        assert "suggestion" in g
        assert "icon" in g
    print("  [PASS] All 9 classes have complete sorting guidance")


def main():
    print("=" * 60)
    print("EcoSort AI — Smoke Tests")
    print("=" * 60)

    tests = [
        test_config_imports,
        test_model_creation,
        test_model_save_load,
        test_preprocessing,
        test_prediction_pipeline,
        test_invalid_image_handling,
        test_transforms,
        test_sorting_guidance_completeness,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            print(f"\nRunning {test.__name__}...")
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
