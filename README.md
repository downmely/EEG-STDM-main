# EEG Data Augmentation for Depression Detection Using Spatial-Temporal Diffusion Model

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org/)

Official implementation of the paper ​**"EEG Data Augmentation for Depression Detection Using Spatial-Temporal Diffusion Model"** (to be published).

> ​**Note**: The full code will be released upon paper acceptance. This repository currently serves as a project preview.

## 📌 Overview
This repository contains:
- ​**Spatial-Temporal Diffusion Model** for EEG signal augmentation
- ​**Depression detection framework** using augmented EEG data
- Preprocessing pipelines for clinical EEG datasets

## 🚀 Key Features
✔️ Novel diffusion-based EEG data augmentation  
✔️ Joint spatial-temporal feature modeling  
✔️ Depression classification with interpretability analysis  
✔️ Support for multiple EEG formats (EDF, MAT, etc.)

## 📂 Dataset Requirements
The model is designed for:
- Multi-channel EEG recordings (minimum 16 channels recommended)
- Clinical depression datasets with diagnostic labels
- Time-series data with sampling rate ≥ 128Hz

## ⚙️ Installation (Preview)
The following dependencies will be required:
```bash
pip install torch==1.10.0 numpy scipy mne matplotlib
