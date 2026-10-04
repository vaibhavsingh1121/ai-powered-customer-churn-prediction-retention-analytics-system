# AI-Powered Customer Churn Prediction & Retention Analytics System

> A machine-learning-based customer churn prediction and retention analytics system that identifies customers at risk of churn and presents insights through an interactive Streamlit application.

## 📌 Overview

Customer churn is a major business problem, particularly for subscription-based organizations. This project uses the **IBM Telco Customer Churn dataset** to build an end-to-end predictive analytics prototype combining data preparation, exploratory analysis, machine learning, evaluation, visualization, and deployment.

The system provides:
- Customer churn prediction
- Churn probability
- Risk classification
- Executive analytics dashboard
- Individual customer prediction
- Batch CSV prediction
- Business-oriented churn insights

The current implementation is a **decision-support prototype**, not an autonomous retention engine.

## 🎯 Objectives

1. Analyze customer data and identify patterns associated with churn.
2. Prepare numerical and categorical variables for machine learning.
3. Train and compare classification models.
4. Tune and select a suitable churn prediction model.
5. Evaluate the model using Accuracy, Precision, Recall, F1-score and ROC-AUC.
6. Predict individual and batch customer churn risk.
7. Provide an interactive analytics dashboard.
8. Generate business-oriented retention insights.

## 📊 Dataset

**IBM Telco Customer Churn Dataset**

- Total records: **7,043**
- Model input features: **19**
- Target: `Churn`
- Customer ID excluded from modeling

| Status | Count | Percentage |
|---|---:|---:|
| No Churn | 5,174 | 73.46% |
| Churn | 1,869 | 26.54% |
| **Total** | **7,043** | **100%** |

### Numerical Features

`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`

### Categorical Features

`gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

## 🔄 Workflow

```text
IBM Telco Dataset
       ↓
Data Inspection & Cleaning
       ↓
Exploratory Data Analysis
       ↓
Feature / Target Separation
       ↓
80:20 Stratified Train-Test Split
       ↓
Numerical & Categorical Preprocessing
       ↓
Model Training & Comparison
       ↓
Tuned Random Forest
       ↓
Evaluation
       ↓
Joblib Pipeline
       ↓
Streamlit Application
       ↓
Dashboard / Individual Prediction / Batch Prediction
       ↓
Risk Classification & Business Insights
```

## 🤖 Machine Learning

The problem is formulated as binary classification:

```text
0 → No Churn
1 → Churn
```

### Train-Test Split

- Training: **5,634**
- Testing: **1,409**
- Split: **80:20**
- Stratified
- `random_state = 42`

### Models Evaluated

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 73.81% | 50.43% | 78.34% | 61.36% |
| Random Forest | 75.30% | 52.40% | 75.94% | 62.01% |
| **Tuned Random Forest / Saved Pipeline** | **76.65%** | **54.56%** | **71.93%** | **62.05%** |

### Final Model

The saved application model is a tuned **Random Forest** using:

```text
class_weight = balanced
max_depth = 10
min_samples_split = 5
n_jobs = -1
random_state = 42
```

## 📈 Final Results

| Metric | Result |
|---|---:|
| **Accuracy** | **76.65%** |
| **Precision** | **54.56%** |
| **Recall** | **71.93%** |
| **F1-Score** | **62.05%** |
| **ROC-AUC** | **0.8386** |

### Confusion Matrix

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| **Actual No Churn** | 811 | 224 |
| **Actual Churn** | 105 | 269 |

The model correctly detected **269 of 374 actual churners** in the recorded test evaluation.

## 🔎 Feature Importance

Prominent recorded model features include:

1. `tenure`
2. `TotalCharges`
3. `Contract – Month-to-month`
4. `MonthlyCharges`
5. `Contract – Two year`
6. `OnlineSecurity – No`
7. `TechSupport – No`
8. `InternetService – Fiber optic`
9. `PaymentMethod – Electronic check`
10. `Contract – One year`

These are model associations and should not be interpreted as proof of causality.

## ⚖️ Threshold Analysis

| Threshold | Precision | Recall | F1-Score |
|---:|---:|---:|---:|
| 0.30 | 44.11% | 90.11% | 59.23% |
| 0.40 | 48.69% | 84.49% | 61.78% |
| 0.50 | 52.61% | 75.40% | 61.98% |
| 0.60 | 56.12% | 64.97% | 60.22% |
| 0.70 | 65.58% | 54.01% | 59.24% |

Lower thresholds increase recall but also increase false positives. The appropriate threshold should ultimately depend on business costs.

## 🖥️ Streamlit Application

### Executive Dashboard

Provides:
- Total customers
- Churned customers
- Churn rate
- Average monthly charges
- Segment-level analysis
- Payment-method analysis
- Risk analysis
- Business visualizations

### Individual Prediction

A user enters customer attributes and receives:
- Churn prediction
- Churn probability
- Risk category

### Batch Prediction

The application accepts CSV files, validates the input, generates predictions for multiple customers, assigns risk categories, and provides downloadable results.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | ML and preprocessing |
| Matplotlib / Seaborn / Plotly | Visualization |
| Streamlit | Web application |
| Joblib | Model serialization |
| Jupyter Notebook | Model development |
| Git / GitHub | Version control |

## 📁 Suggested Repository Structure

```text
customer-churn-prediction/
│
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── churn_model.pkl
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── predict.py
├── outputs/
│   ├── figures/
│   └── predictions/
├── requirements.txt
├── README.md
└── .gitignore
```

> Adjust this structure to match the actual files in your repository.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app/streamlit_app.py
```

Update the command if your actual Streamlit entry file has a different name.

## 💡 Key Business Insights

The analysis found that:
- Month-to-month contract customers show higher observed churn than longer-contract groups.
- Short-tenure customers are an important segment for retention analysis.
- Electronic-check customers show comparatively high observed churn.
- Customers without Online Security or Technical Support show higher observed churn in the analyzed data.
- Higher-charge customers can be investigated for plan fit and service quality.

These findings are descriptive associations, not causal conclusions.

## 🚧 Current Limitations

- The current implementation relies on a historical Telco churn dataset.
- No live CRM integration.
- No real-time event streaming.
- No production-grade automated retention workflow.
- No causal evaluation of retention interventions.
- No continuous production model monitoring.
- Customer Lifetime Value is not integrated into the current minor implementation.
- Explainable AI is planned as a future enhancement.

## 🚀 Future Scope — Major Project

The minor project provides the foundation for a more advanced customer-retention platform.

```text
Churn Prediction
       ↓
Explainable AI
       ↓
Customer Risk Scoring
       ↓
Customer Segmentation
       ↓
Customer Lifetime Value
       ↓
Retention Recommendation Engine
       ↓
Customer 360 Dashboard
       ↓
Intervention Tracking
       ↓
Retention Outcome Analysis
       ↓
Model Monitoring & Retraining
```

Planned enhancements:
- XGBoost / LightGBM / advanced model benchmarking
- Cross-validation and advanced hyperparameter optimization
- SHAP-based explainability
- Customer Lifetime Value
- Customer segmentation
- Intelligent retention recommendations
- PostgreSQL/database integration
- Customer 360 profiles
- Intervention tracking
- Model and data drift monitoring
- Automated retraining framework
- A/B testing and retention-outcome evaluation

## 👥 Project Team

- **Vaibhav Singh** — 235/UCS/121
- **Vansh Chaudhary** — 235/UCS/123
- **Vishal Hoon** — 235/UCS/126
- **Shivam Sharma** — 235/UCS/112

**Supervisor:** Ms. Snehalata Gautam

**Gautam Buddha University**  
School of Information and Communication Technology  
B.Tech — Computer Science and Engineering  
Academic Session: **2026–2027**

## 📚 Project Deliverables

- Data preprocessing and EDA
- Machine learning model
- Model comparison and evaluation
- Feature importance analysis
- Threshold analysis
- Streamlit application
- Executive dashboard
- Individual prediction
- Batch prediction
- Risk classification
- Project report
- Project presentation

## ⚠️ Disclaimer

This project is intended for academic, analytical and decision-support purposes. A churn prediction represents a probability based on historical patterns and does not guarantee that a specific customer will churn. Business decisions should also consider customer value, intervention cost, service history and business rules.

## ⭐ Project Highlights

```text
✓ 7,043 customer records
✓ 19 predictive input features
✓ 80:20 stratified train-test split
✓ Logistic Regression and Random Forest comparison
✓ Tuned Random Forest final pipeline
✓ 76.65% accuracy
✓ 71.93% recall
✓ 62.05% F1-score
✓ 0.8386 recorded ROC-AUC
✓ Individual churn prediction
✓ Batch CSV prediction
✓ Executive analytics dashboard
✓ Probability-based risk classification
✓ Joblib model pipeline
✓ Streamlit web application
```

## 📌 Project Vision

The long-term vision is to transform the current churn prediction prototype into an end-to-end intelligent customer-retention platform:

> **Predict → Explain → Prioritize → Recommend → Intervene → Measure → Learn**
