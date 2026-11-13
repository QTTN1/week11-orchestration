"""
Pure pandas transformation logic for the CDC WONDER mortality pipeline.

Nothing in this file knows about Airflow. The DAG (cdc_mortality_dag.py) decides
WHEN each function runs; this module decides WHAT each step does. Keeping them
separate means every step can be tested on its own with plain Python.

Data sources (all from CDC WONDER, exported as tab-delimited .txt):
  A. MCD drug/alcohol-induced deaths by cause, 1999-2020 and 2021-2022
  B. UCD drug/alcohol-induced deaths incl. "all other causes", 1999-2020
     (summing every category gives all-cause deaths)
  C. All-cause deaths, 2021-2022 (single-race file)
"""
from __future__ import annotations

import pandas as pd

# One row of the final table = one state x year x race x gender group.
KEYS = ["state", "year", "race", "gender"]

# Bridged race (1999-2020) has 4 groups; Single Race 6 (2021+) has 6.
# Map both onto the 4 bridged groups so the whole timeline is comparable.
RACE_MAP = {
    "American Indian or Alaska Native": "American Indian or Alaska Native",
    "Asian or Pacific Islander": "Asian or Pacific Islander",
    "Asian": "Asian or Pacific Islander",
    "Native Hawaiian or Other Pacific Islander": "Asian or Pacific Islander",
    "Black or African American": "Black or African American",
    "White": "White",
    "More than one race": None,  # no 1999-2020 equivalent -> dropped
}

# CDC drug/alcohol cause codes -> readable column names
CAUSE_SLUG = {
    "D1": "overdose_unintentional",
    "D2": "overdose_suicide",
    "D3": "overdose_homicide",
    "D4": "overdose_undetermined",
    "D9": "other_drug_induced",
    "A1": "alcohol_poisoning",
    "A9": "other_alcohol_induced",
    "O9": "non_drug_alcohol",
}


# --------------------------------------------------------------------------- #
# Extract / standardize
# --------------------------------------------------------------------------- #
def read_wonder(path: str) -> pd.DataFrame:
    """Read a CDC WONDER export. The footer (query notes) lives in the 'Notes'
    column with every data column empty, so keeping rows where Notes is blank
    keeps only real data."""
    df = pd.read_csv(path, sep="\t", dtype=str)
    return df[df["Notes"].isna()].drop(columns="Notes")


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename to snake_case, keep only what we need, and fix types.
    We drop CDC's 'Crude Rate' on purpose: it is text ('Unreliable') for small
    counts, and we recompute rates ourselves from Deaths / Population."""
    rename = {
        "State": "state",
        "Year": "year",
        "Gender": "gender",
        "Race": "race",
        "Single Race 6": "race",
        "Deaths": "deaths",
        "Population": "population",
    }
    cause_col = [c for c in df.columns if c.endswith("Drug/Alcohol Induced Cause Code")]
    if cause_col:
        rename[cause_col[0]] = "cause_code"
    out = df.rename(columns=rename)
    keep = KEYS + (["cause_code"] if cause_col else []) + ["deaths", "population"]
    out = out[keep].copy()
    out["year"] = out["year"].astype(int)
    out["deaths"] = pd.to_numeric(out["deaths"], errors="raise").astype(int)
    out["population"] = pd.to_numeric(out["population"], errors="raise").astype(int)
    return out


def stack(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Vertical combine (UNION ALL): same columns, different years."""
    out = pd.concat(frames, ignore_index=True)
    return out


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
def harmonize_race(df: pd.DataFrame) -> pd.DataFrame:
    """Map single-race groups onto bridged-race groups.

    Mapping 'Asian' and 'Native Hawaiian...' to the same label creates two rows
    with the same key, so we must SUM deaths and population (never average
    rates). Rates are recomputed later from the summed counts."""
    out = df.copy()
    out["race"] = out["race"].map(RACE_MAP)
    unknown = df.loc[~df["race"].isin(RACE_MAP), "race"].unique()
    if len(unknown):
        raise ValueError(f"Unmapped race values: {unknown}")
    out = out.dropna(subset=["race"])
    group_cols = [c for c in out.columns if c not in ("deaths", "population")]
    agg = {"deaths": "sum"}
    if "population" in out.columns:
        agg["population"] = "sum"
    return out.groupby(group_cols, as_index=False).agg(agg)


def pivot_causes(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Long (one row per cause) -> wide (one column per cause).

    Uses .pivot, not .pivot_table, so a duplicate key raises an error instead
    of being silently summed. After this, each KEYS combination is one row,
    which makes the later joins one-to-one.

    Missing cells mean CDC did not release that cell (0-9 deaths are
    suppressed / not shown). We leave them as NaN rather than filling 0."""
    wide = df.pivot(index=KEYS, columns="cause_code", values="deaths")
    wide.columns = [f"{prefix}_{CAUSE_SLUG[c]}_deaths" for c in wide.columns]
    return wide.reset_index()


def ucd_population_and_totals(ucd_long: pd.DataFrame) -> pd.DataFrame:
    """From the UCD file (which covers every cause of death), derive
    population and all-cause deaths per group for 1999-2020.

    Data-quality check: population must be identical across the cause rows of
    a group (it is a property of the group, not the cause)."""
    pop_variants = ucd_long.groupby(KEYS)["population"].nunique()
    if (pop_variants > 1).any():
        raise ValueError("Population differs across causes within a group")
    return ucd_long.groupby(KEYS, as_index=False).agg(
        population=("population", "first"),
        all_cause_deaths=("deaths", "sum"),
    )


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #
def build_final_table(
    mcd_wide: pd.DataFrame,
    ucd_wide: pd.DataFrame,
    ucd_totals: pd.DataFrame,
    allcause_2021: pd.DataFrame,
) -> pd.DataFrame:
    """Combine everything into one row per state x year x race x gender.

    1. Spine = all-cause deaths + population for 1999-2022
       (UCD totals for 1999-2020 STACKED with the 2021-2022 all-cause file).
    2. LEFT JOIN drug/alcohol MCD causes (1999-2022) onto the spine.
    3. LEFT JOIN UCD causes (1999-2020 only -> NaN for 2021-2022).
    validate='one_to_one' makes pandas raise if any key is duplicated, so a
    many-to-many join can never silently multiply rows.
    """
    spine = stack([ucd_totals, allcause_2021.rename(columns={"deaths": "all_cause_deaths"})])
    if spine.duplicated(KEYS).any():
        raise ValueError("Duplicate keys in spine (overlapping years?)")

    out = (
        spine.merge(mcd_wide, on=KEYS, how="left", validate="one_to_one")
        .merge(ucd_wide, on=KEYS, how="left", validate="one_to_one")
    )
    if len(out) != len(spine):
        raise ValueError("Row count changed during merge")

    # Rates per 100,000 recomputed from counts
    out["all_cause_rate"] = out["all_cause_deaths"] / out["population"] * 1e5
    for col in [c for c in out.columns if c.startswith("mcd_") and c.endswith("_deaths")]:
        out[col.replace("_deaths", "_rate")] = out[col] / out["population"] * 1e5

    # Share of all deaths whose UNDERLYING cause was drug/alcohol (1999-2020).
    ucd_drug_alc = [
        c for c in out.columns
        if c.startswith("ucd_") and c.endswith("_deaths") and "non_drug_alcohol" not in c
    ]
    # min_count=1 -> stays NaN for 2021-2022, where no UCD data exists
    out["ucd_drug_alcohol_deaths"] = out[ucd_drug_alc].sum(axis=1, min_count=1)
    out["ucd_drug_alcohol_share"] = out["ucd_drug_alcohol_deaths"] / out["all_cause_deaths"]

    # CDC flags rates from fewer than 20 deaths as statistically unreliable
    out["overdose_rate_reliable"] = out["mcd_overdose_unintentional_deaths"] >= 20

    return out.sort_values(KEYS).reset_index(drop=True)
