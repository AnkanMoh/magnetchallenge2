```md
MagNet Challenge 2 — Global LSTM (ΔH) Submission Pipeline

This repository contains a **submission-ready** pipeline for MagNet Challenge 2 that:
1. Computes (or loads) **global normalization statistics**
2. Fine-tunes a **global LSTM model** that predicts **ΔH** (next-step change in H)
3. Evaluates performance on **Pretest** (prints a per-material summary)
4. Generates the **final Result.zip** in the required format (includes `Parameter_Size.csv`)

---

##Repository Structure

Recommended structure (you may rename folders, just keep paths consistent):

```

.
├── main.py
├── artifacts/
│   ├── baseline_model.pt              # pretrained checkpoint (input)
│   ├── finetuned_model.pt             # produced after fine-tune (output)
│   └── global_norm_stats.json         # produced/loaded (output)
├── data/
│   ├── train/                         # training CSVs
│   ├── pretest/                       # pretest h5 files
│   └── final/                         # final evaluation dataset
└── outputs/
└── Result/                        # submission CSVs are written here

```

---

## Data Layout Requirements

### 1) Training Data (`TRAIN_ROOT`)
Expected structure:
```

TRAIN_ROOT/
3C90/
3C90_1_B.csv
3C90_1_H.csv
...
3C90_7_B.csv
3C90_7_H.csv
3C94/
...
N87/

```

### 2) Pretest (`PRETEST_ROOT`)
Expected structure:
```

PRETEST_ROOT/
3C90/
3C90_Testing_True.h5   (or) 3C90_Testing_Padded.h5
...
N87-2/
N87-2_Testing_True.h5  (or) N87-2_Testing_Padded.h5

```

> `N87-2` is treated as an alias of `N87`.

### 3) Final Evaluation (`FINAL_EVAL_ROOT`)
Expected structure:
```

FINAL_EVAL_ROOT/
Testing/
Material A/
A_Padded_H_seq.csv
A_True_B_seq.csv
A_True_T.csv
Material B/
Material C/
Material D/
Material E/

````

---

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy pandas torch h5py tqdm
````

---

## Configuration

This code is configured via environment variables (preferred for clean submissions).

**Minimum required variables:**

* `TRAIN_ROOT`
* `PRETEST_ROOT`
* `FINAL_EVAL_ROOT`
* `START_CHECKPOINT`

Example:

```bash
export TRAIN_ROOT="./data/train"
export PRETEST_ROOT="./data/pretest"
export FINAL_EVAL_ROOT="./data/final"
export START_CHECKPOINT="./artifacts/baseline_model.pt"
```

Optional outputs (defaults shown):

```bash
export STATS_CACHE_PATH="./artifacts/global_norm_stats.json"
export FINETUNE_OUT="./artifacts/finetuned_model.pt"
export FINAL_RESULT_DIR="./outputs/Result"
```

Optional run controls:

```bash
export DEVICE="cpu"                 # "cpu" or "cuda"
export FORCE_RECOMPUTE_STATS="0"    # set "1" to recompute global stats
export SEED="42"
```

---

## Run

Execute the full pipeline:

```bash
python main.py
```

What you should see:

* Global stats loaded/saved
* Fine-tuning progress for `EXTRA_EPOCHS`
* Pretest summary printed per material
* `outputs/Result/` populated with:

  * `A_Pred_H_seq.csv ... E_Pred_H_seq.csv`
  * `Parameter_Size.csv`
* A `Result.zip` created at:

  * `outputs/Result.zip`

---

## Submission Output

The script creates:

* `outputs/Result/Parameter_Size.csv`
* `outputs/Result/A_Pred_H_seq.csv`
* `outputs/Result/B_Pred_H_seq.csv`
* `outputs/Result/C_Pred_H_seq.csv`
* `outputs/Result/D_Pred_H_seq.csv`
* `outputs/Result/E_Pred_H_seq.csv`
* `outputs/Result.zip` (containing the `Result/` folder)

Upload **Result.zip** to the competition submission portal.



