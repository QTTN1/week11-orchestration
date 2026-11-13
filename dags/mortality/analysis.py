"""
Analysis steps that run AFTER the cleaned data is loaded into Postgres:
  1. a visualization of overdose-death trends
  2. a simple baseline ML model

The same target definition (make_model_frame) is reused by the Part 2 agents,
so the Airflow baseline and the agent workflow are scored on the same task.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # no display inside a container
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["state", "race", "gender"]
NUMERIC = ["year"]
TARGET = "high_overdose"


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def pooled_rate(df: pd.DataFrame, deaths_col: str, by: list[str]) -> pd.DataFrame:
    """National rate = total deaths / total population (per 100k).
    Averaging state rates would weight Wyoming the same as California."""
    d = df.dropna(subset=[deaths_col])
    g = d.groupby(by, as_index=False)[[deaths_col, "population"]].sum()
    g["rate"] = g[deaths_col] / g["population"] * 1e5
    return g


def make_charts(df: pd.DataFrame, out_path: str) -> str:
    trend = pooled_rate(df, "mcd_overdose_unintentional_deaths", ["year", "race", "gender"])
    share = df.dropna(subset=["ucd_drug_alcohol_deaths"]).groupby(
        ["year", "race"], as_index=False
    )[["ucd_drug_alcohol_deaths", "all_cause_deaths"]].sum()
    share["pct"] = share["ucd_drug_alcohol_deaths"] / share["all_cause_deaths"] * 100

    races = sorted(trend["race"].unique())
    fig, axes = plt.subplots(2, len(races), figsize=(4.2 * len(races), 7.5), sharex=True)
    for i, race in enumerate(races):
        ax = axes[0, i]
        for gender, color in [("Male", "#2a6fdb"), ("Female", "#d9468f")]:
            s = trend[(trend.race == race) & (trend.gender == gender)]
            ax.plot(s.year, s.rate, marker="o", ms=3, color=color, label=gender)
        ax.set_title(race, fontsize=10)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel("Unintentional overdose deaths\nper 100k (MCD)")
            ax.legend(frameon=False)

        ax2 = axes[1, i]
        s = share[share.race == race]
        ax2.bar(s.year, s.pct, color="#6b7280")
        ax2.grid(alpha=0.3, axis="y")
        ax2.set_xlabel("Year")
        if i == 0:
            ax2.set_ylabel("% of all deaths with drug/alcohol\nunderlying cause (UCD)")

    fig.suptitle("Drug and alcohol mortality in the US, 1999-2022 (CDC WONDER)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Baseline model
# --------------------------------------------------------------------------- #
def make_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, float]:
    """Task: classify whether a state x year x race x gender group has a HIGH
    unintentional overdose death rate (above the median of reliable groups).

    Only groups with >= 20 overdose deaths are used, because CDC considers
    rates from fewer deaths unreliable."""
    d = df[df["overdose_rate_reliable"].astype(bool)].copy()
    threshold = float(d["mcd_overdose_unintentional_rate"].median())
    y = (d["mcd_overdose_unintentional_rate"] > threshold).astype(int).rename(TARGET)
    X = d[CATEGORICAL + NUMERIC]
    return X, y, threshold


def baseline_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ])
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))])


def train_baseline(df: pd.DataFrame) -> dict:
    X, y, threshold = make_model_frame(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = baseline_pipeline().fit(X_tr, y_tr)
    pred = model.predict(X_te)
    return {
        "model": "LogisticRegression (one-hot state/race/gender + year)",
        "target": f"overdose rate > median ({threshold:.1f} per 100k)",
        "n_rows": int(len(X)),
        "n_test": int(len(X_te)),
        "majority_class_accuracy": round(float(max(y_te.mean(), 1 - y_te.mean())), 4),
        "accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "f1": round(float(f1_score(y_te, pred)), 4),
    }


# --------------------------------------------------------------------------- #
# Hand-off to the Part 2 agent loop
# --------------------------------------------------------------------------- #
MODEL_COLUMNS = [
    "state", "year", "race", "gender", "population", "all_cause_rate",
    "mcd_alcohol_poisoning_rate", "mcd_other_alcohol_induced_rate",
    "mcd_other_drug_induced_rate", "mcd_overdose_suicide_rate",
    "mcd_overdose_undetermined_rate",
    # Columns built from the SAME deaths as the target. They are published on
    # purpose (a realistic wide table) and listed in the lineage file so the
    # agents' leakage auditor knows to drop them.
    "mcd_overdose_unintentional_deaths", "mcd_overdose_unintentional_rate",
    "ucd_drug_alcohol_share",
]
DERIVED_FROM_TARGET = [
    "mcd_overdose_unintentional_deaths",  # deaths / population = the target's rate
    "mcd_overdose_unintentional_rate",    # the target is a threshold on this column
    "ucd_drug_alcohol_share",             # its numerator includes the overdose deaths
]


def publish_model_dataset(df: pd.DataFrame, out_dir: str, name: str = "mortality_model") -> dict:
    """Write the modeling table + a lineage file for the agent workflow."""
    from pathlib import Path

    _, y, threshold = make_model_frame(df)
    model_df = df.loc[y.index, MODEL_COLUMNS].copy()
    model_df[TARGET] = y
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{name}.csv"
    model_df.to_csv(csv_path, index=False)
    lineage = {
        "target": TARGET,
        "target_definition": f"mcd_overdose_unintentional_rate > {threshold:.2f} (median of reliable groups)",
        "derived_from_target": DERIVED_FROM_TARGET,
        "source": "cdc_mortality_pipeline (Airflow) -> Postgres table mortality_by_group",
    }
    save_json(lineage, str(out / f"{name}.lineage.json"))
    return {"csv": str(csv_path), "rows": int(len(model_df)), "target": TARGET}


def save_json(obj: dict, path: str) -> str:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path
