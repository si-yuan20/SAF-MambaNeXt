# SAF-MambaNeXt

**SAF-MambaNeXt: Uncertainty-Guided Heterogeneous Collaborative Learning for Reliable white blood cell Classification in Clinical Diagnostic Workflows **

This repository will provide the official implementation of **SAF-MambaNeXt**, Uncertainty-Guided Heterogeneous Collaborative Learning for Reliable white blood cell Classification in Clinical Diagnostic Workflows, integrating:
- ConvNeXt for local morphological feature extraction,
- Mamba (State Space Model) for long-range dependency modeling,
- Structure-Aided Attention Fusion (SAF) for edge-guided cross-branch interaction,
- Uncertainty-Guided Bilateral Fusion (UGBF) for reliability-aware decision fusion.

📌 **Status**:  
The code and processed datasets will be released **after the acceptance of the paper**.

---

## 📄 Paper

If you use this work, please cite our paper:

> SAF-MambaNeXt: Uncertainty-Guided Heterogeneous Collaborative Learning for Reliable white blood cell Classification in Clinical Diagnostic Workflows 
> Rong Gao, Qi Ke, Aiquan Li, Xingning Qin, Sichao Zhao  
> **, 2026 (under review)

(Official citation and DOI will be updated after acceptance.)

---

## 🚀 Planned Contents

After acceptance, this repository will include:

- ✅ Full training and evaluation code (PyTorch)
- ✅ Implementation of:
  - ConvNeXt branch
  - Mamba branch
  - SAF module
  - UGBF module
- ✅ Data preprocessing and augmentation scripts
- ✅ Configuration files for experiments
- ✅ Pretrained model weights
- ✅ Reproduction scripts for all main results in the paper
- ✅ Instructions for training, testing, and visualization (heatmaps, confusion matrices, etc.)

---

## 📊 Datasets

We will provide scripts to prepare and use the following public datasets:

- **PBC** (Peripheral Blood Cell Dataset)
- **LDWBC**
- **Raabin-WBC**

⚠️ Due to dataset licenses, raw data will **not** be redistributed here.  
We will provide:
- Download links to official sources  
- Preprocessing scripts  
- Dataset split files (train/val/test)

---

## 🛠️ Environment (Planned)

- Python >= 3.8  
- PyTorch >= 1.13  
- CUDA (recommended)  
- Additional dependencies will be listed in `requirements.txt`

---

## 📈 Reproducibility

After release, you will be able to reproduce:

- Overall performance on PBC, LDWBC, Raabin-WBC
- Ablation studies (ConvNeXt / Mamba / SAF / UGBF combinations)
- Computational efficiency analysis
- Visualization results (heatmaps, confusion matrices)

---

## 📦 Usage (Coming Soon)

Examples (to be provided after release):

```bash
# Training
python train.py --config configs/saf_mambanext.yaml

# Evaluation
python test.py --weights checkpoints/saf_mambanext.pth

# Visualization
python visualize.py --input sample.jpg
