# Explainable-Credit-Risk-Model
An end-to-end Explainable AI framework for credit risk prediction using the Kaggle "Give Me Some Credit" dataset. The project integrates advanced ensemble models (XGBoost, LightGBM, Random Forest) with SHAP and LIME explainability, fairness auditing, and regulatory-focused documentation to build transparent and accountable loan default prediction.


Explainable-Credit-Risk-Model/
│
├── 0. Data/                         # Raw dataset (never modified)
│   └── GiveMeSomeCredits/
│       ├── cs-training.csv
│       ├── cs-test.csv
│       ├── Data Dictionary.csv
│       └── sampleEntry.csv
│
├── 1. Data engineering/             # Cleaning & feature engineering
│   ├── DA.ipynb
│   ├── data engineering.docx
│   ├── requirements.txt
│   └── updated_data.csv
│
├── 2. Black_box/                    # Model training & comparison
│   ├── Black_box.ipynb
│   ├── experiment_log.csv
│   ├── lgb_predictions.csv
│   ├── logistic_predictions.csv
│   ├── rf_predictions.csv
│   ├── xgb_predictions.csv
│   ├── model_auc_comparison.xlsx
│   ├── Modelling_Documentation.docx
│   ├── test_data.csv
│   └── updated_data.csv
│
├── 3. Export/                       # Production-ready artifacts
│   ├── lgb_model.joblib
│   ├── logistic_model.joblib
│   ├── rf_model.joblib
│   ├── xgb_model.joblib
│   └── prediction outputs (.csv)
│
├── 4. Explainability/               # Model interpretability
│   ├── explainability.ipynb
│   ├── lime_explanation_customer1.html
│   ├── lime_explanation_customer1.pdf
│   ├── shap_summary_plot.png
│   ├── shap_bar_plot.png
│   ├── shap_force_plot.html
│   └── shap_force_plot.pdf
│
└── 5. Validation/                   # Performance + fairness audit
    ├── Validation&Evaluation.ipynb
    ├── validation_metrics.py
    ├── age_features_engineering.py
    ├── age_fairness_analysis.py
    └── generate_validation_report.py