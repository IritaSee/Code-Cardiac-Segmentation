# TransUNet K-Fold Cross-Validation Training

This document describes how to perform k-fold cross-validation training for the **TransUNet model** with **MIFOCAT loss** in the cardiac segmentation project.

## Overview

The TransUNet implementation provides two interfaces for k-fold training:
1. **Command-line script** (`train_transunet_kfold.py`) - For automated training
2. **Jupyter notebook** (`train_transunet_kfold_notebook.ipynb`) - For interactive exploration

Both use the same underlying infrastructure (`train_kfold_wrapper.py`) and support:
- K-fold cross-validation (default 5 folds)
- MIFOCAT unified loss function
- Automatic checkpointing and early stopping
- Per-fold and aggregated metrics

## Quick Start

### 1. Prerequisites

Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

Key dependencies:
- TensorFlow >= 2.15
- NumPy < 2
- OpenCV
- scikit-learn

### 2. Prepare Data Splits

Generate k-fold metadata (only needed once):

```python
from split_data import CardiacDataSplitter

splitter = CardiacDataSplitter(
    input_folder="path/to/data",
    output_folder="./kfold_results"
)

splitter.kfold_split(n_splits=5, val_ratio=0.1)
```

This creates `kfold_metadata.json` with patient-to-fold assignments.

### 3. Run K-Fold Training

#### Option A: Command-Line Script

```bash
python train_transunet_kfold.py \
    --data-dir "Data Test Resize 128 ED/" \
    --fold-metadata kfold_metadata.json \
    --output-dir transunet_kfold_results \
    --epochs 50 \
    --batch-size 32
```

**Available arguments:**
- `--data-dir`: Base directory with patient folders (required)
- `--fold-metadata`: Path to kfold_metadata.json (required)
- `--output-dir`: Output directory (default: `./transunet_kfold_results`)
- `--epochs`: Maximum epochs per fold (default: 50)
- `--batch-size`: Batch size (default: 32)
- `--early-stop-patience`: Early stopping patience (default: 10)
- `--seed`: Random seed (default: 42)
- `--train-only`: Skip test evaluation
- `--start-fold`: Resume from specific fold

#### Option B: Jupyter Notebook

Open and run `train_transunet_kfold_notebook.ipynb`:

```bash
jupyter lab train_transunet_kfold_notebook.ipynb
```

The notebook provides:
- Step-by-step workflow
- Configuration cells
- Results visualization
- Model inspection

## Model Architecture

The TransUNet implementation (`transunet_model.py`) provides:

### Custom Layers
- `patch_extract`: Extracts image patches for transformer processing
- `patch_embedding`: Embeds patches as transformer tokens
- `SwinTransformerBlock`: Swin Transformer attention blocks
- `WindowAttention`: Window-based multi-head self-attention
- `Mlp`: Multi-layer perceptron blocks
- `GELU`, `Snake`: Custom activation functions

### Model Building

```python
from transunet_model import build_transunet_mifocat

model = build_transunet_mifocat(
    input_shape=(256, 256, 1),
    num_classes=4,
    patch_size=16,
    embed_dim=768,
    num_heads=12
)
```

**Parameters:**
- `input_shape`: Input image shape (default: (256, 256, 1))
- `num_classes`: Number of segmentation classes (default: 4)
- `patch_size`: Patch size for transformer (default: 16)
- `embed_dim`: Embedding dimension (default: 768)
- `num_heads`: Number of attention heads (default: 12)
- `window_size`: Window size for Swin Transformer (default: 7)

### Model Size
- **TransUNet**: ~7.7M parameters
- **U-Net** (baseline): ~31M parameters
- TransUNet is **4x smaller** while incorporating transformer attention

## Output Structure

After training, the output directory contains:

```
transunet_kfold_results/
├── kfold_metadata.json          # Patient-to-fold assignments
├── fold_results.json            # Per-fold metrics
├── aggregated_results.json      # Cross-fold statistics
├── fold_0/
│   ├── fold_0_best_model.h5     # Best model checkpoint
│   └── fold_0_history.json      # Training history
├── fold_1/
│   └── ...
└── fold_4/
    └── ...
```

### Metrics Saved

**Per fold:**
- `final_loss`, `final_val_loss`: Last epoch metrics
- `best_val_loss`: Best validation loss
- `epochs_trained`: Actual epochs before early stopping
- `test_loss`: Test set performance

**Aggregated:**
- `val_loss_mean`, `val_loss_std`: Validation loss statistics
- `test_loss_mean`, `test_loss_std`: Test loss statistics
- `n_folds_completed`: Number of successfully completed folds

## Loading Trained Models

To load a saved TransUNet model:

```python
from keras.models import load_model
from transunet_model import get_custom_objects

# Load with custom objects
model_path = "transunet_kfold_results/fold_0/fold_0_best_model.h5"
model = load_model(model_path, custom_objects=get_custom_objects(), compile=False)

# Use for prediction
predictions = model.predict(test_images)
```

## Integration with Existing Workflow

The TransUNet k-fold implementation mirrors the U-Net workflow (`train_kfold_notebook.ipynb`):

1. **Same data loader**: Uses `FoldAwareDataLoader` from `custom_datagen.py`
2. **Same loss function**: MIFOCAT loss from `proposed_model.py`
3. **Same metrics**: Dice, IoU, accuracy
4. **Same fold structure**: Compatible with existing k-fold metadata

### Switching Between Models

In existing scripts, simply change:

```python
# Before (U-Net)
trainer.run_all_folds(model_type='unet', ...)

# After (TransUNet)
trainer.run_all_folds(model_type='transunet', ...)
```

## Comparison with U-Net

| Feature | U-Net | TransUNet |
|---------|-------|-----------|
| **Architecture** | Pure CNN | CNN + Transformer |
| **Parameters** | ~31M | ~7.7M (4x smaller) |
| **Receptive field** | Local (convolutions) | Global (attention) |
| **Training speed** | Faster | Slower (attention overhead) |
| **Memory usage** | Lower | Higher (attention matrices) |

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python train_transunet_kfold.py ... --batch-size 16
```

### Training Too Slow
- Use GPU if available
- Reduce number of transformer layers
- Use smaller embedding dimension

### Model Loading Errors
Always use `get_custom_objects()` when loading:
```python
from transunet_model import get_custom_objects
model = load_model(path, custom_objects=get_custom_objects())
```

## Next Steps

After k-fold training:

1. **Compute detailed metrics**:
   ```bash
   python hitung_evaluasi_metrik.py --model-dir transunet_kfold_results
   ```

2. **Compare with U-Net**:
   - Statistical significance tests
   - Visualization of predictions
   - Hausdorff distance, Dice, MCC comparison

3. **Generate visualizations**:
   - Per-fold loss curves
   - Prediction overlays
   - Failure case analysis

## References

- Original TransUNet implementation: `predict_trans_unet.py`
- U-Net k-fold workflow: `train_kfold_notebook.ipynb`
- MIFOCAT loss: `proposed_model.py`
- Data loading: `custom_datagen.py`

## Support

For issues or questions:
1. Check that `kfold_metadata.json` exists and is valid
2. Verify data directory structure matches expected format
3. Ensure TensorFlow and dependencies are correctly installed
4. Review training logs in output directory

---

**Author**: ramad  
**Date**: 2026-01-28  
**Version**: 1.0
