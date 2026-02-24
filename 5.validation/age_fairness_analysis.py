"""
Age Fairness Analysis
======================
Performs fairness checks and creates plots based on age groups.
Includes distribution plots, correlation analysis, and per-age-group metrics.

Outputs saved to `Model_Results/` directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score


def plot_age_distribution(df, output_dir='Model_Results'):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style('whitegrid')

    if 'age' not in df.columns:
        raise KeyError("'age' column not found")

    # -------------------------
    # Age histogram
    # -------------------------
    plt.figure(figsize=(7, 4))
    sns.histplot(df['age'].dropna(), bins=30, kde=False, color='C0')
    plt.title('Age distribution')
    plt.xlabel('age')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/age_hist.png', dpi=150)
    plt.close()

    # -------------------------
    # Age vs target
    # -------------------------
    if 'Actual' in df.columns:
        dfp = df.copy()
        dfp['_target_'] = dfp['Actual']

        plt.figure(figsize=(6, 4))
        sns.boxplot(
            x='_target_',
            y='age',
            hue='_target_',        # ✅ Added hue
            data=dfp,
            palette='Set2',
            legend=False           # ✅ Disable duplicate legend
        )
        plt.title('Age by target (boxplot)')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/age_box_by_target.png', dpi=150)
        plt.close()

        # -------------------------
        # Positive rate by age bin
        # -------------------------
        if 'age_bin_fixed' in dfp.columns:
            grp = (
                dfp.groupby('age_bin_fixed')['_target_']
                .agg(['count', 'mean'])
                .rename(columns={'mean': 'positive_rate'})
                .reset_index()
            )

            grp.to_csv(f'{output_dir}/age_bin_summary.csv', index=False)

            plt.figure(figsize=(6, 3))
            sns.barplot(
                x='age_bin_fixed',
                y='positive_rate',
                hue='age_bin_fixed',    # ✅ Added hue
                data=grp,
                palette='Blues',
                legend=False            # ✅ Disable duplicate legend
            )
            plt.ylim(0, 1)
            plt.title('Positive rate by age bin (fixed)')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/age_posrate_by_agebin.png', dpi=150)
            plt.close()

        # -------------------------
        # Correlation
        # -------------------------
        corr = dfp['age'].corr(dfp['_target_'])
        with open(f'{output_dir}/age_target_correlation.txt', 'w') as f:
            f.write(f'Pearson correlation (age vs target): {corr}\n')

    # -------------------------
    # Basic statistics
    # -------------------------
    stats = df['age'].describe()
    stats.to_frame().to_csv(f'{output_dir}/age_stats.csv')

    print(f'Age distribution plots saved to {output_dir}/')


def compute_fairness_by_age(X, y, models, output_dir='Model_Results'):
    os.makedirs(output_dir, exist_ok=True)

    if 'age_bin_fixed' not in X.columns and 'age' not in X.columns:
        raise KeyError("'age' or 'age_bin_fixed' not found — run age feature engineering first.")

    # Ensure age_bin_fixed exists
    if 'age_bin_fixed' not in X.columns:
        X = X.copy()
        X['age_bin_fixed'] = pd.cut(
            X['age'].fillna(-1),
            bins=[-1, 25, 45, 65, 999],
            labels=['0-25', '26-45', '46-65', '66+']
        )

    rows = []

    for name, model in models.items():
        print(f"Computing fairness by age -> {name}")

        if hasattr(model, 'feature_names_in_'):
            expected = list(model.feature_names_in_)
        elif hasattr(model, 'feature_name_'):
            expected = list(model.feature_name_)
        else:
            expected = list(X.columns)

        Xm = X.copy()

        for c in expected:
            if c not in Xm.columns:
                Xm[c] = 0

        Xm = Xm[expected].fillna(0)

        # Predictions
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(Xm)[:, 1]
        else:
            probs = model.predict(Xm).astype(float)

        preds = (probs >= 0.5).astype(int)

        dfg = pd.DataFrame({
            'group': X['age_bin_fixed'],
            'y_true': y.values if hasattr(y, 'values') else y,
            'y_pred': preds
        })

        for g, sub in dfg.groupby('group'):
            n = len(sub)
            pos_rate = sub['y_pred'].mean()

            tp = ((sub['y_true'] == 1) & (sub['y_pred'] == 1)).sum()
            fn = ((sub['y_true'] == 1) & (sub['y_pred'] == 0)).sum()
            fp = ((sub['y_true'] == 0) & (sub['y_pred'] == 1)).sum()
            tn = ((sub['y_true'] == 0) & (sub['y_pred'] == 0)).sum()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
            precision = precision_score(sub['y_true'], sub['y_pred'], zero_division=0)
            recall = recall_score(sub['y_true'], sub['y_pred'], zero_division=0)

            rows.append({
                'model': name,
                'age_group': str(g),
                'n': n,
                'positive_rate': pos_rate,
                'tpr': tpr,
                'fpr': fpr,
                'precision': precision,
                'recall': recall
            })

    if rows:
        df_fair_age = pd.DataFrame(rows)
        df_fair_age.to_csv(f'{output_dir}/fairness_by_age.csv', index=False)

        for m, sub in df_fair_age.groupby('model'):
            sub.to_csv(f'{output_dir}/{m}_fairness_by_age.csv', index=False)

        print(f'\nFairness metrics saved to {output_dir}/')
        return df_fair_age
    else:
        print("No rows generated for fairness analysis")
        return None


if __name__ == '__main__':
    df = pd.read_csv('updated_data.csv')

    if 'Actual' not in df.columns:
        raise ValueError("No 'Actual' column found in updated_data.csv")

    y = df['Actual']
    X = df.drop('Actual', axis=1)

    # Only convert numeric columns to float (safer)
    X = X.apply(pd.to_numeric, errors='ignore')

    plot_age_distribution(df)

    import joblib
    models = {
        "XGBoost": joblib.load("xgb_model.joblib"),
        "LightGBM": joblib.load("lgb_model.joblib"),
        "RandomForest": joblib.load("rf_model.joblib")
    }

    compute_fairness_by_age(X, y, models)
