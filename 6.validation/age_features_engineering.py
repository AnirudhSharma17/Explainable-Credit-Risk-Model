"""
Age Feature Engineering
========================
Creates age-related features (age, age_norm, age_sq, age_bin_fixed, age_bin_q)
from a DataFrame column.

Outputs saved to `Model_Results/age_features.csv`.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime


def engineer_age_features(df, age_col=None, output_dir='Model_Results'):
    """
    Create age-related engineered features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe (modified in place).
    age_col : str, optional
        Name of age/DOB column. If None, auto-searches for age-like column.
    output_dir : str
        Directory to save age features CSV.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with age features added.
    """
    os.makedirs(output_dir, exist_ok=True)
    now_year = pd.Timestamp.now().year
    
    # Find age-like column if not specified
    if age_col is None:
        age_candidates = [c for c in df.columns if any(k in c.lower() for k in ('age','dob','birth','birth_year','year_of_birth'))]
        age_col = next((c for c in age_candidates if c.lower() == 'age'), age_candidates[0] if age_candidates else None)
    
    if age_col is None:
        print("No age-like column found — creating empty 'age' column with NaN")
        df['age'] = np.nan
    else:
        val = df[age_col]
        if 'dob' in age_col.lower() or 'date' in age_col.lower():
            df['age'] = pd.to_datetime(val, errors='coerce').dt.year
            df['age'] = now_year - df['age']
        elif 'birth' in age_col.lower() or 'year' in age_col.lower():
            by = pd.to_numeric(val, errors='coerce')
            if by.dropna().le(120).all():
                df['age'] = by
            else:
                df['age'] = now_year - by
        else:
            df['age'] = pd.to_numeric(val, errors='coerce')
    
    # Create engineered columns
    if 'age_norm' not in df.columns:
        mean_age = df['age'].mean(skipna=True)
        std_age = df['age'].std(skipna=True)
        std_age = std_age if std_age and std_age != 0 else 1.0
        df['age_norm'] = (df['age'] - mean_age) / std_age
    
    if 'age_sq' not in df.columns:
        df['age_sq'] = df['age'] ** 2
    
    # Fixed human-friendly buckets
    if 'age_bin_fixed' not in df.columns:
        df['age_bin_fixed'] = pd.cut(df['age'].fillna(-1), bins=[-1,25,45,65,999], labels=['0-25','26-45','46-65','66+'])
    
    # Quartile bins
    if 'age_bin_q' not in df.columns:
        try:
            df['age_bin_q'] = pd.qcut(df['age'].fillna(df['age'].median()), q=4, labels=False, duplicates='drop')
        except Exception:
            df['age_bin_q'] = pd.cut(df['age'].fillna(-1), bins=4, labels=False)
    
    # Save age features
    cols_to_save = [c for c in ('age','age_norm','age_sq','age_bin_fixed','age_bin_q') if c in df.columns]
    df[cols_to_save].to_csv(f'{output_dir}/age_features.csv', index=False)
    
    print(f"Age features created. Non-null age count: {df['age'].notna().sum()}")
    print(f"Saved to {output_dir}/age_features.csv")
    
    return df


if __name__ == '__main__':
    # Example usage
    df = pd.read_csv('updated_data.csv')
    engineer_age_features(df)
