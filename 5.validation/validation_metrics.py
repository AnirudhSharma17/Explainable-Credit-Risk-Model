"""
Validation & Evaluation Metrics
================================
Computes performance metrics, confusion matrices, ROC/PR/calibration plots,
bootstrap stability estimates, and VIF for all models.

Outputs saved to `Model_Results/` directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve


def run_validation(X_eval, y_eval, models, output_dir='Model_Results'):
    """
    Run evaluation for all models.
    
    Parameters:
    -----------
    X_eval : pd.DataFrame
        Feature matrix for evaluation.
    y_eval : pd.Series
        Target labels.
    models : dict
        Dictionary of {model_name: model_object}.
    output_dir : str
        Directory to save outputs.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    # coerce numeric
    X_eval_numeric = X_eval.copy()
    for c in X_eval_numeric.columns:
        try:
            X_eval_numeric[c] = pd.to_numeric(X_eval_numeric[c])
        except Exception:
            pass

    def align_features(X_df, model):
        if hasattr(model, 'feature_names_in_'):
            expected = list(model.feature_names_in_)
        elif hasattr(model, 'feature_name_'):
            expected = list(model.feature_name_)
        else:
            expected = list(X_df.columns)
        X_tmp = X_df.copy()
        for col in expected:
            if col not in X_tmp.columns:
                X_tmp[col] = 0
        return X_tmp[expected]

    metrics_list = []
    for name, model in models.items():
        print(f"Evaluating -> {name}")
        Xm = align_features(X_eval_numeric, model)
        Xm = Xm.fillna(0)

        # get predicted probability
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(Xm)
            y_proba = proba[:, 1] if proba.ndim == 2 else proba
        else:
            y_proba = model.predict(Xm).astype(float)

        y_pred = (y_proba >= 0.5).astype(int)

        # summary metrics
        acc = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred, zero_division=0)
        rec = recall_score(y_eval, y_pred, zero_division=0)
        f1 = f1_score(y_eval, y_pred, zero_division=0)
        roc = roc_auc_score(y_eval, y_proba) if len(np.unique(y_eval)) == 2 else None
        brier = brier_score_loss(y_eval, y_proba) if len(np.unique(y_eval)) == 2 else None
        ll = log_loss(y_eval, y_proba) if len(np.unique(y_eval)) == 2 else None

        metrics = {
            'model': name,
            'n_samples': len(y_eval),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc,
            'brier': brier,
            'logloss': ll
        }
        metrics_list.append(metrics)
        pd.DataFrame([metrics]).to_csv(f"{output_dir}/{name}_metrics_summary.csv", index=False)

        # confusion matrix
        cm = confusion_matrix(y_eval, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} — Confusion matrix")
        plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.savefig(f"{output_dir}/{name}_confusion_matrix.png", bbox_inches='tight', dpi=150)
        plt.close()

        # ROC / PR / calibration (binary only)
        if len(np.unique(y_eval)) == 2:
            fpr, tpr, _ = roc_curve(y_eval, y_proba)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC {roc:.3f}")
            plt.plot([0,1],[0,1],'k--', alpha=0.3)
            plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f"{name} — ROC")
            plt.legend()
            plt.savefig(f"{output_dir}/{name}_roc.png", bbox_inches='tight', dpi=150)
            plt.close()

            precs, recs, _ = precision_recall_curve(y_eval, y_proba)
            pr_auc = auc(recs, precs)
            plt.figure()
            plt.plot(recs, precs, label=f"PR AUC {pr_auc:.3f}")
            plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f"{name} — Precision-Recall")
            plt.legend()
            plt.savefig(f"{output_dir}/{name}_pr.png", bbox_inches='tight', dpi=150)
            plt.close()

            prob_true, prob_pred = calibration_curve(y_eval, y_proba, n_bins=10)
            plt.figure()
            plt.plot(prob_pred, prob_true, 'o-', label='Reliability')
            plt.plot([0,1],[0,1],'k--', alpha=0.3)
            plt.xlabel('Mean predicted probability'); plt.ylabel('Fraction of positives'); plt.title(f"{name} — Calibration")
            plt.legend()
            plt.savefig(f"{output_dir}/{name}_calibration.png", bbox_inches='tight', dpi=150)
            plt.close()

        # classification report
        clf_rep = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)
        pd.DataFrame(clf_rep).transpose().to_csv(f"{output_dir}/{name}_classification_report.csv")

    print(f'\nAll evaluation artifacts saved under `{output_dir}/`')
    metrics_df = pd.DataFrame(metrics_list).set_index('model')
    print(metrics_df)
    return metrics_df


if __name__ == '__main__':
    # Example usage
    import joblib
    
    # Load sample data
    df = pd.read_csv('updated_data.csv')
    if 'Actual' in df.columns:
        y_eval = df['Actual']
        X_eval = df.drop('Actual', axis=1)
    else:
        raise ValueError("No 'Actual' column found in updated_data.csv")
    
    X_eval = X_eval.astype(float)
    
    # Load models
    models = {
        "XGBoost": joblib.load("xgb_model.joblib"),
        "LightGBM": joblib.load("lgb_model.joblib"),
        "RandomForest": joblib.load("rf_model.joblib")
    }
    
    run_validation(X_eval, y_eval, models)
