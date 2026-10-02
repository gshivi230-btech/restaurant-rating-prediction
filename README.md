# 🍽️ Restaurant Rating Prediction Using Machine Learning

A supervised machine learning regression project to predict a restaurant's aggregate rating based on features such as location, cuisines, price range, and service availability.

---

## 📌 Project Overview

- **Problem Type:** Supervised Machine Learning — Regression
- **Objective:** Predict a restaurant's `Aggregate rating` using various restaurant-related attributes.
- **Dataset:** 9,551 rows and 21 columns containing restaurant characteristics and customer feedback.
- **Target Variable:** `Aggregate rating` (numerical value ranging from 0.0 to 5.0).

Customer ratings play a significant role in helping people choose where to dine, but ratings are influenced by many subtle factors—from the variety of cuisines offered to price points and whether online ordering or table bookings are available. In this project, I built an end-to-end regression workflow using Python and Scikit-learn to explore the relationship between these attributes and the final aggregate rating, comparing three different machine learning models to see how well they estimate ratings on unseen test data.

---

## 📊 Dataset Description

The dataset includes **9,551 restaurant records** with **21 columns**:

| Column Name | Description |
| :--- | :--- |
| `Restaurant ID` | Unique identifier for each restaurant record |
| `Restaurant Name` | Name of the restaurant |
| `Country Code` | Numeric code indicating the restaurant's country |
| `City` | City where the restaurant is located |
| `Address` | Complete street address of the restaurant |
| `Locality` | Specific locality or neighborhood |
| `Locality Verbose` | Detailed locality description including city |
| `Longitude` | Geographic longitude coordinate |
| `Latitude` | Geographic latitude coordinate |
| `Cuisines` | Types of food/cuisines served by the restaurant |
| `Average Cost for two` | Estimated dining cost for two individuals |
| `Currency` | Currency used for pricing |
| `Has Table booking` | Categorical indicator for table reservation availability |
| `Has Online delivery` | Categorical indicator for online ordering availability |
| `Is delivering now` | Categorical indicator for current delivery status |
| `Switch to order menu` | Categorical flag for switch menu option |
| `Price range` | Price tier categorization (1 to 4) |
| `Aggregate rating` | **Target variable** — overall customer rating |
| `Rating color` | Color coding assigned to the rating band |
| `Rating text` | Descriptive rating summary (e.g., Good, Very Good, Poor) |
| `Votes` | Total count of customer reviews/votes received |

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Feature Preparation
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Final Analysis
```

---

## 🧹 Data Preprocessing & Preparation

Before building predictive models, the dataset underwent several exploration, cleaning, and transformation steps:

1. **Dataset Loading:** Loaded the raw tabular data into Python using Pandas.
2. **Exploratory Data Analysis (EDA):** Inspected basic properties using `.shape`, `.columns`, `.dtypes`, `.describe()`, missing-value summaries, and duplicate checks.
3. **Handling Missing Values:** Detected missing entries in the `Cuisines` column and imputed them with `"Unknown"` to retain those rows.
4. **Duplicate Verification:** Checked for duplicated rows across the dataset.
5. **Preventing Target Leakage:** Removed `Rating color` and `Rating text`. These columns directly describe the rating tier (such as "Good" or "Poor"), which would leak information about the target variable into the input features.
6. **Dropping Identifiers:** Removed `Restaurant ID` because arbitrary identification numbers carry no predictive relationship with ratings.
7. **Removing High-Cardinality Text:** Excluded `Restaurant Name`, `Address`, and `Locality Verbose`. These fields contain unique text values with high cardinality that were not used in the final regression models.
8. **Feature and Target Separation:** Split data into input features (`X`) and target vector (`y = Aggregate rating`).
9. **Categorical Encoding:** Applied Scikit-learn's `OneHotEncoder` to encode categorical columns into numeric binary vectors.
10. **Preprocessing Pipeline:** Encapsulated feature transformations within a Scikit-learn `Pipeline` and `ColumnTransformer` to ensure consistent data handling without leakage.
11. **Train-Test Split:** Partitioned the dataset into:
    - **80% Training Data**
    - **20% Testing Data**
12. **Reproducibility:** Used `random_state=42` across splits and model instances for reproducible results.

---

## 🤖 Models Used

Three supervised regression models were evaluated:

1. **Linear Regression:**
   A simple regression baseline that assumes a straight-line linear relationship between the input features and the aggregate rating. It is useful as a reference point to understand if linear assumptions are sufficient.

2. **Decision Tree Regressor:**
   A non-linear model that partitions the data through a series of decision-based splits (if-else conditions) on feature values. It can capture non-linear relationships and interactions without needing feature scaling.

3. **Random Forest Regressor:**
   An ensemble method that builds an array of multiple decision trees on bootstrap samples of the data and averages their individual predictions. Averaging predictions across trees helps reduce variance compared to a single decision tree.

---

## 📏 Evaluation Metrics

To evaluate model performance on the 20% test split, four standard regression metrics were computed:

- **MAE (Mean Absolute Error):** Measures the average absolute difference between the actual and predicted ratings. Lower MAE indicates predictions are closer to real values on average.
- **MSE (Mean Squared Error):** Measures the average of the squared errors between actual and predicted ratings. It penalizes larger deviations more heavily. Lower MSE is better.
- **RMSE (Root Mean Squared Error):** The square root of MSE, expressing the prediction error in the original units of the rating (0.0 to 5.0). Lower RMSE is better.
- **R² (Coefficient of Determination):** Measures the proportion of variance in the target rating explained by the model features. Values closer to 1.0 indicate that a larger portion of rating variance is accounted for by the model.

---

## 📈 Model Performance & Comparison

The exact test-set evaluation results for each model are summarized below:

| Model | MAE | MSE | RMSE | R² |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.890069 | 1.199810 | 1.095358 | 0.472868 |
| **Decision Tree Regressor** | 0.217552 | 0.110011 | 0.331680 | 0.951667 |
| **Random Forest Regressor** | **0.200635** | **0.094553** | **0.307495** | **0.958458** |

### Key Observations:
- **Random Forest Regressor produced the strongest test-set performance** among the three tested models across all four evaluation metrics, achieving the lowest MAE (~0.20), MSE (~0.09), and RMSE (~0.31), alongside the highest R² score (~0.958).
- **Linear Regression struggled** with an R² of ~0.473 and an MAE of ~0.890, indicating that the relationship between restaurant attributes and aggregate rating has non-linear components that a simple linear model cannot fully capture.
- **Decision Tree Regressor performed strongly** (R² ~ 0.952), and the Random Forest ensemble further refined this by reducing variance, shaving MAE down to ~0.201.
- *Note:* While Random Forest produced solid performance on this test split, these results reflect this specific dataset and feature set; they do not imply that the model is perfect or without error.

---

## 🔍 Detailed Analysis & Findings

The project notebook includes several diagnostic analyses to better understand model behavior:

- **Actual vs. Predicted Ratings Visualization:** Scatter plots comparing actual test ratings against model predictions showed that the Random Forest predictions cluster tightly along the diagonal reference line.
- **Prediction Error Analysis:** Histograms of residuals (actual − predicted) were plotted to examine the distribution of errors, verifying that errors are predominantly centered around zero without large systematic bias.
- **Random Forest Feature Importance:** Inspected feature importance scores from the fitted Random Forest model to observe which input features (such as `Votes`) contributed most significantly to the decision splits.
- **Sample Predictions:** Generated a side-by-side comparison of actual vs. predicted values for individual test records to examine where predictions were accurate and where slight deviations occurred.
- **Model Comparison:** Evaluated all three models under identical conditions to quantify the performance gap between linear and tree-based methods.

---

## 🛠️ Technologies Used

- **Python** — Core programming language
- **Pandas** — Data manipulation, inspection, and cleaning
- **NumPy** — Numerical operations and array handling
- **Scikit-learn** — Preprocessing pipelines, OneHotEncoder, train-test splitting, regression models, and evaluation metrics
- **Matplotlib** — Data visualizations (actual vs. predicted plots, error distributions, feature importance charts)
- **Jupyter Notebook / VS Code** — Interactive development environment

---

## 💡 What I Learned

Working through this project provided practical experience with several key aspects of applied machine learning:

- **Working with a real-world dataset:** Learning how tabular data behaves outside textbook examples, including mixed data types and real-world distributions.
- **Exploratory Data Analysis (EDA):** Using statistical summaries and inspection methods to understand column cardinality, value distributions, and data balance.
- **Data Cleaning & Missing-Value Handling:** Handling missing entries systematically (such as imputing missing cuisine categories with `"Unknown"`) rather than blindly dropping data.
- **Understanding Target Leakage:** Recognizing why variables like `Rating text` and `Rating color` had to be removed—because they are derived directly from the rating, including them would artificially inflate performance without providing real predictive value.
- **Feature Preparation & Pipelines:** Setting up Scikit-learn pipelines with `ColumnTransformer` and `OneHotEncoder` to keep data preprocessing modular, reproducible, and leak-free.
- **Train/Test Splitting:** Splitting data properly with a fixed random seed (`random_state=42`) to ensure honest evaluation on unseen data.
- **Regression Evaluation:** Developing a clear understanding of the differences between MAE, MSE, RMSE, and R², particularly how RMSE penalizes occasional large errors more than MAE.
- **Comparing Different ML Models:** Understanding why tree-based models significantly outperformed linear regression on this dataset by handling non-linear interactions effectively.
- **Random Forest Feature Importance:** Gaining insight into how ensemble trees rank feature relevance and which restaurant attributes drive predictions most.

---

## 📁 Project Structure

```text
restaurant-rating-prediction/
│
├── README.md
├── restaurant_rating_prediction.ipynb
├── model_comparison_results.csv        # Generated model comparison metrics
└── random_forest_predictions.csv       # Generated test predictions
```

*(Note: CSV outputs are generated by running the notebook.)*

---

## 🚀 How to Run

Follow these simple steps to run the project locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/gshivi230-btech/restaurant-rating-prediction.git
   cd restaurant-rating-prediction
   ```

2. **Install the required Python libraries:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```

3. **Open the Jupyter Notebook in Jupyter or VS Code:**
   ```bash
   jupyter notebook restaurant_rating_prediction.ipynb
   ```

4. **Place the dataset in the expected location:**
   Ensure the dataset file (`Dataset .csv`) is placed in the project root directory alongside the notebook.

5. **Run the notebook cells in order:**
   Execute all cells sequentially from top to bottom to run the exploration, preprocessing, model training, evaluation, and visualizations.

---

## 🔮 Future Improvements

Some practical directions to build upon this project in the future include:

- **Hyperparameter Tuning:** Using `GridSearchCV` or `RandomizedSearchCV` to optimize parameters like `n_estimators`, `max_depth`, and `min_samples_split`.
- **Cross-Validation:** Implementing K-Fold cross-validation to assess model stability across multiple validation folds.
- **Additional Feature Engineering:** Creating derived features (such as cost-per-cuisine ratios or interaction features) to test if they improve accuracy.
- **Testing Additional Regression Models:** Evaluating gradient boosting algorithms or other regression methods.
- **Better Handling of High-Cardinality Categorical Features:** Exploring target encoding or frequency encoding for categories with many unique values.
- **Testing on Unseen Restaurant Data:** Testing model performance on newly collected or external restaurant datasets to verify generalizability across different regions.
- **Model Persistence:** Saving the trained Random Forest pipeline using `joblib` or `pickle` so predictions can be generated without retraining.

---

## 🎯 Conclusion

In this project, three regression models—Linear Regression, Decision Tree Regressor, and Random Forest Regressor—were implemented and evaluated to predict restaurant aggregate ratings. Model performance was measured across four metrics: MAE, MSE, RMSE, and R². Among the tested approaches, the Random Forest Regressor produced the strongest test-set performance, achieving an R² of ~0.958 and an MAE of ~0.201. 

Beyond the numerical results, this project provided a structured, hands-on opportunity to practice the full machine learning workflow, from exploratory analysis and data cleaning to handling target leakage, building preprocessing pipelines, and analyzing prediction errors.
