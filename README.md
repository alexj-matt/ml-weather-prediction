# 🌦️ Predicting Rain in Pully  
*A machine learning project from the Introduction to Machine Learning course at EPFL*  

This project explores **binary weather prediction**: given weather data from **day 1**, predict if it rains on **day 2**.  
We built and compared several machine learning models in Julia, ranging from simple logistic regression to neural networks,  
and submitted predictions to a Kaggle competition hosted for the course.  

📂 **Core code:** [`rain.jl`](rain.jl)  
📊 **Figures:** [`figures/`](figures/)  
📑 **Full report:** [`report.pdf`](report.pdf)  

---

## 🛠️ Approach

- **Data preprocessing:**  
  - Filled missing values with mean imputation  
  - Removed predictors with zero variance  
  - Normalized all predictors for comparability  
  - Generated correlation plots for exploratory analysis  

- **Models explored:**  
  - Logistic classifier (baseline)  
  - Multilayer perceptrons with varying depth, width, optimizers, and activation functions  

- **Evaluation strategy:**  
  - Random seed control for reproducibility (`Random.seed!(10)`)  
  - Manual hyperparameter tuning  
  - Submissions to Kaggle for accuracy evaluation  

---

## 📊 Key Results

- **Logistic classifier** achieved early solid results (accuracy ≈ 0.92).  
- **Neural networks** significantly improved predictions:  
  - Best reproducible model: **3 layers × 64 nodes, ADAMW optimizer, ReLU activation** → accuracy **0.953**  
  - Highest score: **3 layers × 128 nodes, ADAMW optimizer** → accuracy **0.962**, but **not fully reproducible** due to late random seed fix.  

---

## 📂 Explore the Project

- [`rain.jl`](rain.jl): main Julia script for preprocessing, fitting, and predicting  
- [`figures/`](figures/): generated plots from preprocessing and analysis  
- [`report.pdf`](report.pdf): detailed academic report of the project  
- [`docs/README-original.md`](docs/README-original.md): original project README for archival  

⚠️ *Datasets (`res/`) and trained models (`machines/`) are not included due to size.*  

---

## 🙌 Team

This project was completed by **Tim Brunner & Alexander Odermatt**  
as part of the **Introduction to Machine Learning** course at **EPFL**.  

---
