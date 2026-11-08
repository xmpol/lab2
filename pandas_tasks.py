"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    df = pd.read_csv(data.csv_path)

    missing_by_column = {
        col: int(cnt) for col, cnt in df.isna().sum().items()
    }

    adults_over_30_count = int((df["Age"] > 30).sum())

    mean_age_by_pclass = {
        int(pclass): float(mean)
        for pclass, mean in df.groupby("Pclass")["Age"].mean().items()
    }

    survival_rate_by_pclass = {
        int(pclass): float(rate)
        for pclass, rate in df.groupby("Pclass")["Survived"].mean().items()
    }
  
    highest_fares = [float(v) for v in df["Fare"].nlargest(5).tolist()]
    
    return TitanicSummary(
        row_count=int(len(df)),
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares,
    )
