"""
CDC drug/alcohol mortality pipeline (Airflow 2.x, TaskFlow API).

Flow:
                    ┌─ [mcd_drug_alcohol]  extract -> harmonize_race -> pivot_causes ─┐
  check_raw_files ──┼─ [ucd_all_causes]    extract -> harmonize_race -> split ────────┼─> merge -> load_to_postgres ─┬─> make_charts
                    └─ [allcause_2021_22]  extract -> harmonize_race ─────────────────┘                             ├─> train_baseline_model
                                                                                                                    └─> publish_model_dataset (-> Part 2 agents)

- The three TaskGroups do not depend on each other, so the LocalExecutor runs
  them in PARALLEL.
- make_charts, train_baseline_model and publish_model_dataset only need the
  database table, so they also run in parallel.
- Tasks pass FILE PATHS (small strings) through XCom, never DataFrames.
  The data itself is written to data/staging/ as CSV between steps.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow.decorators import dag, task, task_group
from airflow.providers.postgres.hooks.postgres import PostgresHook

from mortality import analysis, transform

DATA = Path("/opt/airflow/data")
RAW = DATA / "raw"
STAGING = DATA / "staging"
OUTPUT = Path("/opt/airflow/output")
AGENT_DATASETS = Path("/opt/airflow/agent_datasets")  # = agents/automl-research-lab/datasets
WAREHOUSE_CONN = "warehouse_db"  # defined by AIRFLOW_CONN_WAREHOUSE_DB in docker-compose
FINAL_TABLE = "mortality_by_group"

RAW_FILES = {
    "mcd_1999": RAW / "mcd_drug_alcohol_1999_2020.txt",
    "mcd_2021": RAW / "mcd_drug_alcohol_2021_2022.txt",
    "ucd": RAW / "ucd_drug_alcohol_allcause_1999_2020.txt",
    "allcause_2021": RAW / "allcause_2021_2022.txt",
}


def _save(df: pd.DataFrame, name: str) -> str:
    STAGING.mkdir(parents=True, exist_ok=True)
    path = STAGING / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _engine():
    return PostgresHook(postgres_conn_id=WAREHOUSE_CONN).get_sqlalchemy_engine()


@dag(
    dag_id="cdc_mortality_pipeline",
    description="CDC WONDER drug/alcohol mortality: ingest -> clean -> merge -> Postgres -> analysis",
    schedule="@monthly",              # CDC WONDER refreshes periodically; re-run monthly
    start_date=datetime(2026, 11, 1),
    catchup=False,                    # don't back-fill runs for past months
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    tags=["week11", "cdc", "etl"],
)
def cdc_mortality_pipeline():

    @task
    def check_raw_files() -> dict:
        """Fail fast if any input is missing, before any parallel work starts."""
        missing = [str(p) for p in RAW_FILES.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing raw files: {missing}")
        return {k: str(v) for k, v in RAW_FILES.items()}

    # ------------------------------------------------------------------ #
    # Branch A: MCD drug/alcohol deaths by cause, 1999-2022
    # ------------------------------------------------------------------ #
    @task_group(group_id="mcd_drug_alcohol")
    def mcd_drug_alcohol(files: dict):

        @task
        def extract(files: dict) -> str:
            frames = [transform.standardize(transform.read_wonder(files[k]))
                      for k in ("mcd_1999", "mcd_2021")]
            # population is re-taken from the all-cause data (see README)
            return _save(transform.stack(frames).drop(columns="population"), "mcd_stacked")

        @task
        def harmonize_race(path: str) -> str:
            return _save(transform.harmonize_race(pd.read_csv(path)), "mcd_harmonized")

        @task
        def pivot_causes(path: str) -> str:
            return _save(transform.pivot_causes(pd.read_csv(path), "mcd"), "mcd_wide")

        return pivot_causes(harmonize_race(extract(files)))

    # ------------------------------------------------------------------ #
    # Branch B: UCD all causes, 1999-2020 (gives population + all-cause deaths)
    # ------------------------------------------------------------------ #
    @task_group(group_id="ucd_all_causes")
    def ucd_all_causes(files: dict):

        @task
        def extract(files: dict) -> str:
            return _save(transform.standardize(transform.read_wonder(files["ucd"])), "ucd_long")

        @task
        def harmonize_race(path: str) -> str:
            return _save(transform.harmonize_race(pd.read_csv(path)), "ucd_harmonized")

        @task
        def split_totals_and_causes(path: str) -> dict:
            df = pd.read_csv(path)
            return {
                "totals": _save(transform.ucd_population_and_totals(df), "ucd_totals"),
                "causes": _save(transform.pivot_causes(df.drop(columns="population"), "ucd"), "ucd_wide"),
            }

        return split_totals_and_causes(harmonize_race(extract(files)))

    # ------------------------------------------------------------------ #
    # Branch C: all-cause deaths, 2021-2022
    # ------------------------------------------------------------------ #
    @task_group(group_id="allcause_2021_22")
    def allcause_2021_22(files: dict):

        @task
        def extract(files: dict) -> str:
            return _save(transform.standardize(transform.read_wonder(files["allcause_2021"])), "allcause_2021")

        @task
        def harmonize_race(path: str) -> str:
            return _save(transform.harmonize_race(pd.read_csv(path)), "allcause_2021_harmonized")

        return harmonize_race(extract(files))

    # ------------------------------------------------------------------ #
    # Merge, load, analyze
    # ------------------------------------------------------------------ #
    @task
    def merge(mcd_path: str, ucd_paths: dict, allcause_path: str) -> str:
        final = transform.build_final_table(
            mcd_wide=pd.read_csv(mcd_path),
            ucd_wide=pd.read_csv(ucd_paths["causes"]),
            ucd_totals=pd.read_csv(ucd_paths["totals"]),
            allcause_2021=pd.read_csv(allcause_path),
        )
        return _save(final, "final_merged")

    @task
    def load_to_postgres(path: str) -> str:
        df = pd.read_csv(path)
        df.to_sql(FINAL_TABLE, _engine(), if_exists="replace", index=False, method="multi", chunksize=1000)
        print(f"Loaded {len(df):,} rows into {FINAL_TABLE}")
        return FINAL_TABLE

    @task
    def make_charts(table: str) -> str:
        df = pd.read_sql_table(table, _engine())  # read back FROM the database
        OUTPUT.mkdir(parents=True, exist_ok=True)
        return analysis.make_charts(df, str(OUTPUT / "overdose_trends.png"))

    @task
    def train_baseline_model(table: str) -> dict:
        df = pd.read_sql_table(table, _engine())
        metrics = analysis.train_baseline(df)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        analysis.save_json(metrics, str(OUTPUT / "baseline_metrics.json"))
        pd.DataFrame([metrics]).to_sql("model_metrics", _engine(), if_exists="replace", index=False)
        print(metrics)
        return metrics

    @task
    def publish_model_dataset(table: str) -> dict:
        """Hand-off to Part 2: modeling CSV + lineage file in the agent app's datasets folder."""
        df = pd.read_sql_table(table, _engine())
        info = analysis.publish_model_dataset(df, str(AGENT_DATASETS))
        print(info)
        return info

    # Wiring: passing one task's output into another creates the dependency.
    files = check_raw_files()
    final_path = merge(
        mcd_drug_alcohol(files),
        ucd_all_causes(files),
        allcause_2021_22(files),
    )
    table = load_to_postgres(final_path)
    make_charts(table)
    train_baseline_model(table)
    publish_model_dataset(table)


cdc_mortality_pipeline()
