# SFRC Strength Predictor

A machine learning-based desktop GUI for predicting the **compressive strength (CS)** and **splitting tensile strength (STS)** of steel fiber-reinforced concrete (SFRC).

The pre-trained BO-XGBoost model (`gui_models.pkl`) is included — **no dataset required to run the GUI**.

---

## Model Performance

| Target | Model | Test R² | Test RMSE |
|--------|-------|---------|-----------|
| CS | BO-XGBoost | 0.9206 | 3.422 MPa |
| STS | BO-XGBoost | 0.8928 | 0.543 MPa |

---

## Screenshot
<img width="763" height="593" alt="aadf9fd295d553c702cee38bfef947e1" src="https://github.com/user-attachments/assets/4ed6c51a-0958-4803-871b-98075dd3f424" />



---

## Installation

```bash
git clone https://github.com/xincheng017/SFRC-Strength-Predictor.git
cd SFRC-Strength-Predictor
pip install -r requirements.txt
python SFRC_GUI_ctk.py
```

Or in Jupyter Notebook:

```python
%run SFRC_GUI_ctk.py
```

---

## Input Parameters

| Parameter | Unit | Training Range |
|-----------|------|---------------|
| Water content | kg/m³ | 123 – 264 |
| Cement content | kg/m³ | 258 – 651 |
| Sand content | kg/m³ | 488 – 1225 |
| Coarse aggregate content | kg/m³ | 356 – 1594 |
| Maximum aggregate size | mm | 10 – 40 |
| Superplasticizer | kg/m³ | 0 – 18.4 |
| Fiber volume fraction (Vf) | % | 0 – 3.0 |
| Fiber diameter (df) | mm | 0 – 1.22 |
| Fiber length (Lf) | mm | 0 – 62 |
| Fiber type | — | Hooked / Crimped / Mill-cut / Straight smooth / Chopped with butt ends / No fiber |

> **Note:** Predictions are most reliable within the training data ranges listed above. Results outside these ranges should be interpreted with caution.

---

## Database

The model was trained on a database of **671 experimental records** compiled from **47 published studies**:

- Compressive strength (CS): n = 431
- Splitting tensile strength (STS): n = 240

The full database is available in the [`data/`](./data) directory, including the master file
(`SFRC_cleaned_datasets_v2.xlsx`), CSV exports of both datasets, and a data dictionary
([`data/README.md`](./data/README.md)). The complete list of the **47 source publications**
is provided in [`data/data_sources.md`](./data/data_sources.md).
---

## Hyperparameter Optimization

Three optimization strategies were systematically compared:

| Algorithm | Trials / Evaluations | CS Best R² | STS Best R² |
|-----------|----------------------|------------|-------------|
| Bayesian Optimization (Optuna TPE) | 150 – 200 trials | **0.9206** | **0.8928** |
| Particle Swarm Optimization (PSO) | 600 evaluations | 0.9183 | 0.8850 |
| Genetic Algorithm (GA) | 720 evaluations | 0.9096 | 0.8711 |

BO-XGBoost achieved the best overall performance on both targets.

---

## Dependencies

| Package | Version |
|---------|---------|
| customtkinter | ≥ 5.2.0 |
| xgboost | ≥ 1.7.0 |
| scikit-learn | ≥ 1.1.0 |
| numpy | ≥ 1.23.0 |
| pandas | ≥ 1.5.0 |
| openpyxl | ≥ 3.0.0 |
| pillow | ≥ 9.0.0 |

---


## License

This project is released for research reference only.  
The pre-trained model weights are provided as-is without warranty of any kind.
