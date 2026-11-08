"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

import pandas as pd

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]

def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])

    for col in FEATURES + ["MRI_Count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    gender = df["Gender"].astype(str).str.strip().str.lower()

    men_df = df[gender == "male"]
    women_df = df[gender == "female"]

    men_corr = {
        feat: float(men_df[feat].corr(men_df["MRI_Count"], method="pearson"))
        for feat in FEATURES
    }
    
    women_corr = {
        feat: float(women_df[feat].corr(women_df["MRI_Count"], method="pearson"))
        for feat in FEATURES
    }

    candidates = {**men_corr, **women_corr}
    strongest_mri_feature = max(candidates, key=lambda k: abs(candidates[k]))

    return BrainCorrelationSummary(
        men_count=int(len(men_df)),
        women_count=int(len(women_df)),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest_mri_feature,
    )
