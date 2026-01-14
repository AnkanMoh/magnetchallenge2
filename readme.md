
```markdown
# MagNet Challenge 2 — Robust Sequence-Level H-Field Prediction

This repository contains our solution for **MagNet Challenge 2**, focused on **sequence-level prediction of magnetic field intensity H(t) from B(t)** under realistic and incomplete test conditions.

The approach combines a **global LSTM model**, **physics-aware training**, and a **robust inference pipeline** that guarantees valid predictions for all test sequences, including those with early missing values (e.g., Material C).

---

## Problem Overview

The MagNet Challenge evaluates the reconstruction of the magnetic field intensity sequence **H(t)** from the magnetic flux density **B(t)**.

Performance is evaluated using **two official sequence-level metrics only**:

- **`seq_err`** — sequence prediction error of H(t)
- **`seq_ene`** — sequence energy error measuring physical consistency via  
  \[
  E = \sum_t H(t) \cdot \frac{dB(t)}{dt}
  \]

A key challenge is that the **final evaluation dataset contains incomplete and NaN-padded sequences**, where standard sequence models fail due to insufficient context.

---

## Method Summary

### Model Architecture
- **Global LSTM** shared across all materials and frequencies
- Input window length: **80**
- Input features per timestep:
  - Magnetic flux density **B(t)**
  - Magnetic field intensity **H(t)**
  - Time derivative **dB/dt**
- Material and frequency embeddings appended to the LSTM output
- Model predicts **ΔH** instead of absolute H for stability

---

### Training Objective (Aligned with Official Metrics)

The model is trained using a **physics-aware loss** that directly targets the competition metrics:

\[
\mathcal{L} = \text{seq\_err} + \lambda \cdot \text{seq\_ene}
\]

where:
- **seq_err** penalizes sequence-level H prediction error
- **seq_ene** penalizes deviation in magnetic energy
- \(\lambda\) controls the strength of the energy regularization

Certain materials (e.g., **3F4, N49**) are weighted more heavily to improve generalization.

---

## Robust Final Evaluation Handling

To handle incomplete test sequences (especially **Material C**), we implement a **robust inference strategy**:

1. Detect the first NaN index in each sequence
2. If the valid prefix is shorter than 80:
   - Apply edge-padding using the earliest valid value
3. Repair internal NaNs using **linear interpolation**
4. Predict H values **sequentially** using ΔH rollouts
5. Apply a safety fallback (carry-forward last valid H) to ensure forward progress

This guarantees:
- **No skipped rows**
- **No NaNs in final predictions**
- Valid computation of **seq_err** and **seq_ene**

---

## Repository Structure

```

.
├── run_magnet2.py        # End-to-end training + inference script
├── README.md             # This file
├── .gitignore

````

**Not included in the repository** (generated at runtime):
- Model checkpoints (`*.pt`)
- Submission files (`Result/`, `Result.zip`)
- Diagnostic logs

---

## How to Run (Kaggle)

1. Attach the official MagNet datasets:
   - Training dataset
   - Final evaluation dataset
2. Open a Kaggle notebook
3. Run:
   ```bash
   python run_magnet2.py
````

4. The script will:

   * Compute global normalization statistics
   * Train from scratch if no checkpoint is found
   * Fine-tune the model
   * Generate `Result.zip` ready for submission

The final submission will be available at:

```
/kaggle/working/Result.zip
```

---

## Reproducibility Notes

* Global normalization statistics are computed once and cached
* Random seeds are fixed for deterministic behavior
* The pipeline automatically falls back to training from scratch if no checkpoint is found

---

## Key Takeaways

* Optimizes **exactly** the two official metrics: `seq_err` and `seq_ene`
* Physics-aware loss improves physical consistency
* Robust inference ensures valid predictions for all test sequences
* Fully compliant with MagNet Challenge rules


\
```
