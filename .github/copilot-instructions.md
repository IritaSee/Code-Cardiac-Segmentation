# Copilot Instructions for MIFOCAT Research Project

## 1. Project Overview & Research Context
This codebase implements the experiments for the research paper: **"MIFOCAT: A Unified Loss Function for Robust Myocardium Segmentation in Short-Axis Cine Cardiac Magnetic Resonance Images"**.

**Core Goal:** Improve myocardium segmentation accuracy in cardiac MRI (ACDC 2017 dataset) using a unified loss function named **MIFOCAT**.

**The MIFOCAT Loss Function:**
It integrates three components to address specific segmentation challenges:
1.  **Mean Squared Error (MSE/MI):** Minimizes pixel-level discrepancies and enforces spatial smoothness.
2.  **Focal Loss (FO):** Mitigates class imbalance by focusing on hard-to-segment regions.
3.  **Categorical Cross-Entropy (CAT):** Strengthens inter-class discrimination.

**Formula:** $L_{uni} = r_1 L_{MI} + r_2 L_{FO} + r_3 L_{CAT}$

## 2. Codebase & File Structure
The project is built using **Python 3.10** and **TensorFlow/Keras**.

### Key Directories
* **`docs/`**: **ALWAYS refer to this folder for context.** It contains the primary research paper (`mifocat_...pdf`), supplementary materials, and theoretical references. If you are unsure about a formula or methodology, check the documents in this folder first.
* **`hausdorff/`**: Contains the custom implementation for Hausdorff distance metrics.

### Key Python Files
* **Data Pipeline:**
    * `load_dataset*.py`: Handling NIfTI files, resizing, and normalization.
    * `slice3dto2d.py`: Logic for slicing 3D volumes into 2D PNGs.
    * `custom_datagen.py`: Custom Keras `imageLoader`.
* **Models:**
    * `predict_unet_2d.py`: U-Net implementation.
    * `predict_trans_unet.py`: TransUNet implementation (includes custom layers like `SwinTransformerBlock`).
* **Metrics:**
    * `hitung_evaluasi_metrik*.py`: Main script for calculating metrics.
    * `hitung_*.py`: Individual metric logic (Dice, IoU, Surface Distance, MCC).

## 3. Instructions for Code Modifications

### General Guidelines
* **Reference the `docs` folder:** Before implementing new features or writing explanations, cross-reference the `docs/` folder to ensure alignment with the research paper's definitions and goals.
* **Language:** Maintain existing variable naming conventions (often a mix of English and Indonesian, e.g., `citra`, `hitung`), but prefer English for new modules.
* **Paths:** The codebase currently uses hardcoded absolute paths. When refactoring, suggest relative paths.

### When creating new metrics:
1.  **Pattern:** Follow the functional style of existing `hitung_*.py` files.
2.  **Input:** Expect `groundtruth` and `predict` numpy arrays (typically 2D or batch).
3.  **Integration:** New metrics must be integrated into `hitung_evaluasi_metrik.py` inside the loop processing each patient/slice.

### When adjusting models:
1.  **Custom Objects:** `predict_trans_unet.py` relies heavily on custom layers. Always include these in the `custom_objects` dictionary when loading models.
2.  **Loss Integration:** Ensure the **MIFOCAT** loss combination (MSE + Focal + CAT) is implemented as a custom loss function block.

## 4. Current Tasks (Reviewer Requests)
The immediate goal is to address reviewer feedback. This likely involves:
1.  **New Metrics:** Adding metrics that measure boundary quality or volumetric consistency.
2.  **Ablation Studies:** Scripting runs that isolate specific loss components ($L_{MI}$, $L_{FO}$, $L_{CAT}$).
3.  **Visualization:** Enhancing scripts to highlight failure cases (red arrows in the paper).