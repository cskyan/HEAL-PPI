# HEAL-PPI — Hierarchical Evidence-Aligned Learning for Protein Interaction Analysis

HEAL-PPI is a hierarchical multimodal framework for protein interaction modeling with **dual outputs**: a fused binary interaction/interface prediction and an evidence-oriented Top-K scoring branch derived from the refined contact field. The public release focuses on the reproducible training and inference pipeline used for the benchmark and downstream analyses reported in our study.

The released code supports four input modalities at the residue level:

- **Sequence embeddings** from ESM-2
- **Structure-derived inputs** from 3D coordinates / contact supervision
- **PSSM** features
- **DSSP** features

The repository also includes the curated **RBP296** downstream dataset used in our task-oriented adaptation experiments. Other public datasets are referenced through official download links, with only the corresponding identifier lists redistributed here.

---

## Highlights

- **Hierarchical evidence-aligned predictor** with residue-level, fragment-level, and contact-level reasoning
- **Adaptive fusion with dual outputs**:
  - `p_fused`: binary interaction / interface probability
  - `evi_score` / `evi_logit`: Top-K evidence-oriented readout
- **Four-modality residue input interface**:
  - sequence
  - structure-derived supervision
  - PSSM
  - DSSP
- **Public-release friendly layout** with placeholder paths and local ESM checkpoint support
- **RBP296 included** for downstream adaptation and interpretation-oriented experiments

---

## Repository Layout

```text
HEAL-PPI/
├─ Data/
│  ├─ RBP296/
│  │  ├─ RBP296_final_ids.txt
│  │  ├─ coords/
│  │  ├─ dssp/
│  │  ├─ labels/
│  │  ├─ pssm/
│  │  └─ seq/
│  ├─ Benchmark5.5/
│  │  └─ DB5.5_ids.txt
│  ├─ DIPS-Plus/
│  │  └─ DIPS-Plus_list.txt
│  ├─ Esm/
│  │  └─ esm2_t33_650M_UR50D.pt.txt
│  └─ Train355Test60/
│     ├─ Train_335_ids.txt
│     └─ Test_60_ids.txt
├─ config_topk.py
├─ loss_innovations.py
├─ model.py
├─ predict.py
├─ train.py
├─ train_sitepairs.py
├─ environment.yml
├─ LICENSE
└─ README.md
```

---

## Environment

A conda environment file is provided:

```bash
conda env create -f environment.yml
conda activate heal-ppi
```

The environment includes the core dependencies required by the current public codebase:

- `pytorch`
- `torchvision`
- `torchaudio`
- `pytorch-cuda`
- `fair-esm`
- `numpy`
- `scikit-learn`
- `biopython`
- `networkx`
- `matplotlib`
- `seaborn`
- `openpyxl`

If your machine uses a different CUDA version, you may need to adjust the PyTorch / CUDA entries in `environment.yml`.

---

## ESM Checkpoint

This repository **does not redistribute** the ESM-2 model weights.

The current pipeline expects a local checkpoint named:

```text
esm2_t33_650M_UR50D.pt
```

Place it under a directory such as:

```text
/path/to/resources/esm/esm2_t33_650M_UR50D.pt
```

Then set:

```bash
export ESM_LOCAL_DIR=/path/to/resources/esm
```

On Windows PowerShell:

```powershell
$env:ESM_LOCAL_DIR = "D:\path\to\resources\esm"
```

By default, the sequence embedder uses a **local** ESM checkpoint through `train_sitepairs.py`.

---

## Data Download and Preparation

### Included in this repository

The following resource is included directly:

- **RBP296** downstream dataset under `Data/RBP296/`

### Public datasets not redistributed

The following public datasets are **not** fully mirrored here. Instead, this repository provides identifier lists and expects users to download the raw resources from their official sources.

#### 1. DIPS / DIPS-Plus

Used as the principal interface-level structural training benchmark.

- Official access: [DIPS](https://github.com/drorlab/DIPS)
- Local identifier list provided here: `Data/DIPS-Plus/DIPS-Plus_list.txt`

#### 2. Protein-Protein Docking Benchmark 5.5 (DB5.5)

Used as the principal cross-dataset L2 evaluation benchmark.

- Official access: [DB5.5 benchmark page](https://zlab.wenglab.org/benchmark/)
- Local identifier list provided here: `Data/Benchmark5.5/DB5.5_ids.txt`

#### 3. Train335 / Test60 benchmark split

Used as the principal residue-level benchmark following the GraphPPIS-style protocol.

- Official access: [GraphPPIS](https://github.com/biomed-AI/GraphPPIS)
- Local identifier lists provided here:
  - `Data/Train355Test60/Train_335_ids.txt`
  - `Data/Train355Test60/Test_60_ids.txt`

---

## RBP296 Data Format

The included `RBP296` dataset is organized as a multimodal single-chain resource compatible with the released scripts.

Expected subdirectories:

- `seq/`: amino-acid sequences in FASTA-like format
- `coords/`: structure-compatible coordinate files and derived tensors
- `pssm/`: residue-wise PSSM features
- `dssp/`: residue-wise DSSP features
- `labels/`: residue-level labels for downstream adaptation / evaluation
- `RBP296_final_ids.txt`: curated full identifier list

The current public scripts treat `RBP296` as a **complete downstream dataset**. During training / evaluation, the code reads the full identifier list and performs a reproducible random split according to the configuration.

---

## Split Protocol

For the released `RBP296` pipeline, the split is generated automatically from the full ID list using the configuration in `config_topk.py`:

- `split_train`
- `split_val`
- `split_test`
- `split_seed`

The prediction script first looks for a saved `split_ids.json` in the run directory. If the file is not found, it recomputes the split from the full ID list using the same settings.

This means the repository does **not require** separate fixed train / val / test text files for `RBP296`, although users may optionally save and reuse `split_ids.json` for exact reproducibility.

---

## Quickstart

### 1. Configure paths

Edit `config_topk.py` or override paths through environment variables.

Typical placeholders include:

- `RBP_ROOT`
- `RBP_ID_LIST`
- `ESM_LOCAL_DIR`
- `SPLIT_TRAIN`
- `SPLIT_VAL`
- `SPLIT_TEST`
- `SPLIT_SEED`

Example on Linux:

```bash
export RBP_ROOT=/path/to/HEAL-PPI/Data/RBP296
export RBP_ID_LIST=/path/to/HEAL-PPI/Data/RBP296/RBP296_final_ids.txt
export ESM_LOCAL_DIR=/path/to/resources/esm
```

Example on Windows PowerShell:

```powershell
$env:RBP_ROOT = "E:\path\to\HEAL-PPI\Data\RBP296"
$env:RBP_ID_LIST = "E:\path\to\HEAL-PPI\Data\RBP296\RBP296_final_ids.txt"
$env:ESM_LOCAL_DIR = "D:\path\to\resources\esm"
```

### 2. Train

```bash
python train.py
```

### 3. Predict / Evaluate

```bash
python predict.py --split test
```

---

## Outputs

The public release preserves the dual-output design used in our framework.

### Output 1: Binary prediction

- `p_fused`
- Used for binary interaction / interface probability estimation

### Output 2: Evidence prediction

- `evi_score`
- `evi_logit`
- Derived from the refined contact/evidence field and used for Top-K evidence analysis

The prediction pipeline can also export ranking-ready files, dense maps, and Top-K pair summaries for downstream analysis.

---

## Reproducibility Notes

- The complete experimental setup is controlled primarily through `config_topk.py`
- Public benchmark datasets other than `RBP296` are referenced through official download links
- `RBP296` is included for downstream adaptation and interpretation-oriented experiments
- ESM weights are not redistributed and must be downloaded separately
- Exact benchmark preprocessing for public datasets should follow their official resources and the conventions described in our manuscript / supplementary materials

---

## Citation

If you use this repository in academic work, please cite the corresponding paper once the final bibliographic entry is available.

```bibtex
@article{healppi,
  title   = {HEAL-PPI: Hierarchical Evidence-Aligned Learning for Protein Interaction Analysis},
  author  = {To be updated},
  journal = {To be updated},
  year    = {To be updated}
}
```

---

## License

This project is released under the license provided in [LICENSE](LICENSE).
