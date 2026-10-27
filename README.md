# 🍽️ Cognifyz Restaurant Rating Prediction System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Cognifyz Technologies Data Science Internship — Level 1 / Level 2 Task: Restaurant Rating Prediction**  
> An end-to-end Machine Learning web application and regression pipeline predicting restaurant aggregate ratings based on location, cuisines, pricing, and service offerings with **95.85% R² accuracy**.

---

## 🌐 Live Streamlit Web Application

The interactive web dashboard is ready to deploy on **Streamlit Community Cloud**:
- **Interactive Predictor:** Real-time rating estimates with gauge chart, rating tiers, and confidence intervals.
- **Model Comparison:** Live evaluation metrics across Linear Regression, Decision Tree, and Random Forest.
- **Dataset Explorer:** Search and filter 9,551 restaurants across 141 cities.

---

## 📋 Project Overview

- **Problem Type:** Supervised Machine Learning — Regression
- **Objective:** Accurately predict a restaurant's `Aggregate rating` (0.0 to 5.0 scale) using restaurant attributes.
- **Dataset:** 9,551 rows and 21 columns containing restaurant characteristics and customer review data.
- **Target Variable:** `Aggregate rating`
- **Best Model:** **Random Forest Regressor** (R² = **0.9585**, MAE = **0.2006**, RMSE = **0.3075**)

Customer ratings play a crucial role in dining choices. This project builds an end-to-end regression pipeline using Python, Scikit-learn, and Streamlit to evaluate the impact of various restaurant features on customer ratings.

---

## 📊 Dataset Description

The dataset consists of **9,551 restaurant records** with **21 attributes**:

| Column Name | Description |
| :--- | :--- |
| `Restaurant ID` | Unique record identifier (dropped — non-predictive) |
| `Restaurant Name` | Restaurant brand name (dropped — high cardinality) |
| `Country Code` | Numeric country identifier |
| `City` | City where the restaurant is located |
| `Address` | Street address (dropped) |
| `Locality` | Neighborhood / area |
| `Locality Verbose` | Detailed locality description (dropped) |
| `Longitude` | Geographic longitude coordinate |
| `Latitude` | Geographic latitude coordinate |
| `Cuisines` | Food types and cuisines served |
| `Average Cost for two` | Estimated dining cost for two people |
| `Currency` | Pricing currency |
| `Has Table booking` | Table reservation availability (`Yes`/`No`) |
| `Has Online delivery` | Online delivery availability (`Yes`/`No`) |
| `Is delivering now` | Current delivery active status (`Yes`/`No`) |
| `Switch to order menu` | Switch menu option |
| `Price range` | Price tier categorization (1: Budget to 4: Luxury) |
| `Aggregate rating` | **Target Variable** (0.0 to 5.0) |
| `Rating color` | Rating band color (dropped — target leakage) |
| `Rating text` | Descriptive rating summary (dropped — target leakage) |
| `Votes` | Total count of customer reviews/votes |

---

## ⚙️ Data Preprocessing & Pipeline

1. **Target Leakage Prevention:** Removed `Rating color` and `Rating text` to prevent direct leakage into the model.
2. **Identifier & High-Cardinality Pruning:** Dropped `Restaurant ID`, `Restaurant Name`, `Address`, and `Locality Verbose`.
3. **Missing Value Imputation:** Imputed missing values in `Cuisines` with `"Unknown"`.
4. **Encoding Pipeline:** Used Scikit-learn `ColumnTransformer` with `OneHotEncoder(handle_unknown="ignore")` for categorical features and passthrough for numeric features.
5. **Split:** 80% Training split and 20% Testing holdout split (`random_state=42`).

---

## 🏆 Model Evaluation & Comparison

| Model | MAE | MSE | RMSE | R² Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.8901 | 1.1998 | 1.0954 | 0.4729 |
| **Decision Tree Regressor** | 0.2176 | 0.1100 | 0.3317 | 0.9517 |
| **Random Forest Regressor (Champion)** | **0.2006** | **0.0946** | **0.3075** | **0.9585** |

### Key Findings:
- **Random Forest** achieved the highest accuracy with **95.85% R²**, reducing MAE to **0.20** rating points.
- **Review Volume (`Votes`)** and **Price Range / Cost** are the strongest predictors of restaurant ratings.

---

## 🚀 Streamlit Web Application Features

1. **Rating Predictor Tab:**
   - Smart city auto-fill: Selecting a city automatically adjusts Country Code, Currency, Coordinates, and Localities.
   - Dynamic multi-select cuisine selection.
   - Real-time gauge visualization with official color-coded rating tiers.
   - Contextual insight breakdown for predicted scores.
2. **Model Performance Tab:**
   - Interactive bar charts comparing MAE, RMSE, and R² scores across models.
   - Top 12 feature importance ranking from Random Forest.
   - Actual vs. Predicted scatter plot on test holdout data.
3. **Dataset Explorer Tab:**
   - Multi-criteria restaurant filtering by City, Rating range, and Price tier.
   - "Test Random Record" button that tests live model predictions against actual ground-truth ratings.

---

## 💻 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gshivi230-btech/restaurant-rating-prediction.git
   cd restaurant-rating-prediction
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with your GitHub account.
3. Click **"New app"**.
4. Select repository: `gshivi230-btech/restaurant-rating-prediction`
5. Branch: `main`
6. Main file path: `app.py`
7. Click **"Deploy!"**. Your live URL will be generated within seconds!

---

## 📂 Repository Structure

```plaintext
restaurant-rating-prediction/
├── app.py                             # Interactive Streamlit Web Dashboard
├── requirements.txt                   # Production Python dependencies
├── model.joblib                       # Pretrained Random Forest model bundle
├── Dataset .csv                       # Clean restaurant dataset
├── model_comparison_results.csv       # Benchmark metrics across models
├── random_forest_predictions.csv      # Test split prediction results
├── restaurant_rating_prediction.ipynb # Complete EDA & training notebook
├── .gitignore                         # Standard git ignore configuration
└── README.md                          # Comprehensive documentation
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
