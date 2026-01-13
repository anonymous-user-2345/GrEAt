# GrEAt — Generalizable Energy-based Anomaly Detection (Tabular)

GrEAt is a dual-head transformer framework for anomaly detection in **tabular** data. It learns (1) a supervised classifier for known anomalies and (2) a scalar **energy** function that separates normal/noisy samples from distribution-shifted anomalies using margin-based objectives and triplet regularization. The model is trained in **two phases**: classification pretraining, then energy-based refinement. 

---

## ✨ Highlights

* **Energy-optimized detection.** Low energy for in-distribution & benign covariate shifts; high energy for semantically novel anomalies that disrupt inter-attribute dependencies. 
* **Two-phase training.** Phase 1: weighted cross-entropy; Phase 2: margin losses (normal/anomaly) + triplet loss for robustness to noise. 
* **Tabular-ready Transformer.** Mixed numeric/categorical inputs (scaling + embeddings) with a [CLS] summary passed to classification & energy heads. 
* **Strong results across domains.** Outperforms 8 ML/DL baselines on 6 benchmarks (biomedical, finance, network security); up to **+0.19 F1** on known anomalies and **+0.29 F1** on unseen semantic anomalies. 

---

## 📦 What’s in this repo

* `GrEAt.ipynb` — end-to-end GrEAt training pipeline (two-phase training, energy margins, triplet loss, evaluation).
* `baseLine.ipynb` — baseline implementations (IF, OCSVM, LOF, AE, MLP, LSTM, Transformer, DeepSVDD) and comparison harness.

> The notebooks mirror the paper’s methodology and experiments, including dataset loading, preprocessing, model training, and metric reporting. 

---

## 🔧 Installation

```bash
# Create a fresh environment (conda recommended)
conda create -n great python=3.10 -y
conda activate great

# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn pandas numpy matplotlib notebook
```

> CPU training reproduces small/medium datasets; a GPU is helpful for million-row datasets like KDD-Cup (paper used an NVIDIA L4 for that experiment). 

---

## 🚀 Quickstart

1. Launch Jupyter:

   ```bash
   jupyter notebook
   ```
2. Open **`GrEAt.ipynb`** to train/evaluate GrEAt.
3. (Optional) Open **`baseLine.ipynb`** to train baselines and compare.

---

## 🧠 Method (at a glance)

* **Inputs:** numeric columns are MinMax-scaled; categorical columns use learnable embeddings; a learnable `[CLS]` token summarizes each row. The `[CLS]` embedding feeds:

  * **Classification head** (binary logits, weighted cross-entropy),
  * **Energy head** (scalar energy; hinge margins: (m_n<0) for normal/noisy, (m_a>0) for anomalies) + **triplet loss** with Gaussian-noised positives. 
* **Training:**

  * **Phase 1:** ( \lambda_{\text{cls}}=1 ) (classification only).
  * **Phase 2:** activate energy margins & triplet with ( \lambda_{\text{cls}}+\lambda_n+\lambda_a+\lambda_t=1 ). 

---

## 📚 Datasets

Experiments cover six real-world tabular datasets across biomedical, finance, and network security (sizes & anomaly rates as used in the paper): Vertebral, Ecoli, Breast Cancer Wisconsin, German Credit (Statlog), Thyroid, and **KDD-Cup** (~976k rows, 41 features; 0.35% anomaly rate). 

---

## 📊 Key Results

* **Known anomalies (Objective 2):**
  GrEAt attains top or near-top scores on all datasets, e.g., **Ecoli** (Acc **0.94**, F1 **0.97**), **Vertebral** (Acc **0.81**, F1 **0.88**), **Breast Cancer** (Acc **0.97**, F1 **1.00**), **Thyroid** (F1 **0.92**, within 0.02 Acc of best). See Table 2 in the paper for full comparisons. 
* **Unseen semantic anomalies vs noise (Objective 3):**
  Under mutation analysis (Type-1, Type-2, Gaussian noise), GrEAt either leads or matches the best across datasets; notably **German Credit** (F1 **0.82/0.82/0.83** for A1/A2/A3), and **Ecoli** (F1 **0.90/0.88/0.97**). See Table 3. 
* **Scalability (Objective 4):**
  Sub-quadratic scaling on CPU for small/medium datasets (e.g., Thyroid ~669s). KDD-Cup trained in ~3,780s on an L4 GPU. See Table 4. 

---

## 📁 Data access

Links are provided in the Datasets directory. 

---

## 🧪 Reproducing the results

* Use the **two-phase** schedule as in the notebook.
* Follow the **loss-weight grid** in Phase 2 (the paper fixes (\lambda_{\text{cls}}=0.5) and distributes the remaining 0.5 among (\lambda_n,\lambda_a,\lambda_t) from {0.1, 0.2, 0.25}). 
* Evaluate with **Accuracy, Precision, Recall, F1**, and (for robustness) **mutation analysis** (A1/A2/A3). 

---

## 🧾 Citation

If you use GrEAt, please cite the paper:

```bibtex
@article{2026GrEAt,
  title   = {GrEAt: Generalizable Energy-based Anomaly Detection in Tabular Data},
  author  = {Anonymous},
  year    = {2025}
}
```

