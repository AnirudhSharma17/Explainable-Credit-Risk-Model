"""
Fairness & Validation Summary Report Generator
=============================================
Consolidates all model validation, performance, and fairness findings into
a comprehensive report (text + HTML + visualizations).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# ==============================
# Load Results
# ==============================

def load_results(output_dir='Model_Results'):
    metrics_list = []
    fairness_df = None
    age_stats = None

    for fname in os.listdir(output_dir):
        if 'metrics_summary.csv' in fname:
            df = pd.read_csv(os.path.join(output_dir, fname))
            metrics_list.append(df)

    fairness_path = os.path.join(output_dir, 'fairness_by_age.csv')
    if os.path.exists(fairness_path):
        fairness_df = pd.read_csv(fairness_path)

    age_stats_path = os.path.join(output_dir, 'age_stats.csv')
    if os.path.exists(age_stats_path):
        age_stats = pd.read_csv(age_stats_path, index_col=0)

    metrics_df = (
        pd.concat(metrics_list, ignore_index=True).set_index('model')
        if metrics_list else None
    )

    return metrics_df, fairness_df, age_stats


# ==============================
# Fairness Detection
# ==============================

def detect_fairness_issues(fairness_df):
    issues = {}

    if fairness_df is None:
        return issues

    for model in fairness_df['model'].unique():
        sub = fairness_df[fairness_df['model'] == model]
        flags = []

        tpr_vals = sub['tpr'].dropna()
        if len(tpr_vals) > 1:
            gap = tpr_vals.max() - tpr_vals.min()
            if gap > 0.05:
                flags.append(f"⚠️ TPR Gap: {gap:.1%}")

        fpr_vals = sub['fpr'].dropna()
        if len(fpr_vals) > 1:
            gap = fpr_vals.max() - fpr_vals.min()
            if gap > 0.05:
                flags.append(f"⚠️ FPR Gap: {gap:.1%}")

        pos_vals = sub['positive_rate'].dropna()
        if len(pos_vals) > 1:
            gap = pos_vals.max() - pos_vals.min()
            if gap > 0.10:
                flags.append(f"⚠️ Positive Rate Gap: {gap:.1%}")

        issues[model] = flags if flags else ["✅ No major fairness issues detected"]

    return issues


# ==============================
# Text Report
# ==============================

def generate_text_summary(metrics_df, fairness_df, age_stats):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = []

    lines.append("=" * 90)
    lines.append("FAIRNESS & VALIDATION SUMMARY REPORT".center(90))
    lines.append("=" * 90)
    lines.append(f"Generated: {timestamp}".center(90))
    lines.append("")

    if metrics_df is not None:
        lines.append("MODEL PERFORMANCE")
        lines.append("-" * 90)
        for model, row in metrics_df.iterrows():
            lines.append(f"\n{model.upper()}")
            lines.append(f"  Accuracy:  {row['accuracy']:.1%}")
            lines.append(f"  Precision: {row['precision']:.1%}")
            lines.append(f"  Recall:    {row['recall']:.1%}")
            lines.append(f"  F1-Score:  {row['f1']:.1%}")
            lines.append(f"  ROC-AUC:   {row['roc_auc']:.1%}")

    fairness_issues = detect_fairness_issues(fairness_df)
    if fairness_issues:
        lines.append("\nFAIRNESS CHECK")
        lines.append("-" * 90)
        for model, flags in fairness_issues.items():
            lines.append(f"\n{model}")
            for flag in flags:
                lines.append(f"  {flag}")

    lines.append("\n" + "=" * 90)
    lines.append("END OF REPORT".center(90))
    lines.append("=" * 90)

    return "\n".join(lines)


# ==============================
# HTML Report
# ==============================

def generate_html_summary(metrics_df, fairness_df):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Validation Report</title>
        <style>
            body {{ font-family: Arial; background:#f4f6f7; padding:20px; }}
            .container {{ background:white; padding:20px; border-radius:8px; }}
            table {{ width:100%; border-collapse:collapse; margin-top:10px; }}
            th, td {{ padding:10px; border-bottom:1px solid #ddd; }}
            th {{ background:#2c3e50; color:white; }}
        </style>
    </head>
    <body>
        <div class="container">
        <h1>Model Validation & Fairness Report</h1>
        <p>Generated: {timestamp}</p>
    """

    if metrics_df is not None:
        html += "<h2>Model Performance</h2><table>"
        html += "<tr><th>Model</th><th>Accuracy</th><th>ROC-AUC</th></tr>"
        for model, row in metrics_df.iterrows():
            html += f"""
            <tr>
                <td>{model}</td>
                <td>{row['accuracy']:.1%}</td>
                <td>{row['roc_auc']:.1%}</td>
            </tr>
            """
        html += "</table>"

    html += "</div></body></html>"

    return html


# ==============================
# Visualizations
# ==============================

def create_comparison_visualizations(fairness_df, output_dir):
    if fairness_df is None:
        return

    sns.set_style("whitegrid")

    plt.figure(figsize=(8, 5))
    pivot = fairness_df.pivot(index='age_group', columns='model', values='tpr')
    pivot.plot(kind='bar', ax=plt.gca())
    plt.title("TPR by Age Group")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fairness_tpr.png"), dpi=150)
    plt.close()


# ==============================
# MAIN REPORT FUNCTION
# ==============================

def generate_report(output_dir='Model_Results', report_name='Validation_Report'):
    print("Loading results...")
    metrics_df, fairness_df, age_stats = load_results(output_dir)

    print("Generating text report...")
    text_summary = generate_text_summary(metrics_df, fairness_df, age_stats)
    text_path = os.path.join(output_dir, f"{report_name}_Summary.txt")

    # ✅ FIX: UTF-8 encoding added
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text_summary)

    print("Generating HTML report...")
    html_summary = generate_html_summary(metrics_df, fairness_df)
    html_path = os.path.join(output_dir, f"{report_name}_Summary.html")

    # ✅ FIX: UTF-8 encoding added
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_summary)

    print("Creating visualizations...")
    create_comparison_visualizations(fairness_df, output_dir)

    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETE".center(80))
    print("=" * 80)


if __name__ == "__main__":
    generate_report()
