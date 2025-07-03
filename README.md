# MSc Thesis Script Repository – Decoding the Developing Brain

This repository contains all scripts and documentation for the Master's thesis project:

**"Decoding the Developing Brain: An EEG-Based Functional Connectivity Analysis in Pediatric Multiple Sclerosis"**  
by *Ilse de Wit*  
TU Delft in collaboration with Erasmus MC Child Brain Lab  
Submission: July 2025

---

## 🧠 Project Summary

This thesis investigates whether EEG-based functional connectivity (FC) can serve as a non-invasive biomarker for visual processing speed in children with multiple sclerosis (MS). 
Resting-state EEG recordings were analyzed using a custom pipeline including quality control, source-space connectivity estimation, and validation steps.

---

## 🗂 Repository Structure

![image](https://github.com/user-attachments/assets/137cb047-c31d-4f27-a69a-1e79f4156e22)

### 🔧 Main Scripts

| Script                               | Description                                                                                               |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `Preprocessing.py`                   | Runs preprocessing of raw EEG data including filtering, ICA, and artifact rejection.                      |
| `Functions_load_data.py`             | Helper functions for loading EEG files into the MNE framework.                                            |
| `Functions_quality_metrics.py`       | Functions to compute segment-wise quality metrics (e.g., noise, amplitude, correlation).                  |
| `Validation_preprocessing.py`        | Validates the performance of preprocessing using quality metrics and segment tracking.                    |
| `Validation_quality_assessment.py`   | Validates the automated quality assessment by correlation with a reference                                |
| `Functions_concept_preprocessing.py` | Early experimental preprocessing functions.                                                               |
| `FC_analysis.py`                     | Performs source reconstruction and computes functional connectivity (PLI) and network analysis (MST).     |
| `Quality_check_FC_Berger.py`         | Validates FC output by assessing expected changes in alpha connectivity (Berger effect).                  |
| `Statistical_analysis.py`            | Correlates FC metrics with visual processing speed (Processing Speed Index, WISC-V).                      |

### 📁 File Dependencies

- `.fif` preprocessed EEG files → used by `FC_analysis.py`
- `.npz` result files from `FC_analysis.py` → used by `Quality_check_FC_Berger.py` and `Statistical_analysis.py`

---

## 💻 How to Run

1. Run `Preprocessing.py` to clean EEG data. Adjust input paths as needed.
2. Use `Validation_preprocessing.py` to evaluate preprocessing performance.
3. Run `FC_analysis.py` to compute connectivity metrics.
4. Use `Quality_check_FC_Berger.py` for validation based on the Berger effect.
5. Run `Statistical_analysis.py` to compute correlations with behavioral data.

---

## 📦 Requirements

- Python ≥ 3.9
- MNE-Python
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn
- NetworkX
