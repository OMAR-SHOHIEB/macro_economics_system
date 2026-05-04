from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow.decorators import dag, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "include"))

from pipeline.scraping.world_bank import WorldBankScraper
from pipeline.scraping.imf import get_exchange_rate_panel
from pipeline.processing.clean import EconomicImputer
from pipeline.processing.feature_extraction import Feature_Extraction
from pipeline.processing.feature_selection import Feature_Selection
from pipeline.processing.prepare_data import Handling, Modeling
from pipeline.processing.transform import Make_Transform


@dag(
    dag_id="macro_data_pipeline",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["macro", "pipeline", "etl", "scraping"],
)
def macro_data_pipeline():
    @task(task_id="scrape_world_bank")
    def scrape_world_bank() -> pd.DataFrame:
        wb = WorldBankScraper()
        data_frames = [
            wb.get_gdp_total(),
            wb.get_population(),
            wb.get_gdp_growth(),
            wb.get_gdp_per_capita(),
            wb.get_inflation_rate(),
            wb.get_unemployment_rate(),
            wb.get_fdi(),
            wb.get_exports(),
            wb.get_imports(),
            wb.get_gov_spending(),
            wb.get_investment(),
            wb.get_government_debt(),
            wb.get_interest_rate(),
            wb.get_life_expectancy(),
        ]

        frames = [df for df in data_frames if df is not None and not df.empty]
        merged = pd.concat(frames, ignore_index=True)
        pivot_df = merged.pivot_table(
            index=["Country", "Year"],
            columns="Metric",
            values="Value",
        ).reset_index()
        pivot_df.columns.name = None
        return pivot_df

    @task(task_id="scrape_exchange_rate")
    def scrape_exchange_rate() -> pd.DataFrame:
        df = get_exchange_rate_panel(None)
        return df.rename(columns={"country": "Country", "year": "Year", "exchange_rate": "Exchange Rate"})

    @task(task_id="merge_sources")
    def merge_sources(world_bank_df: pd.DataFrame, imf_df: pd.DataFrame) -> pd.DataFrame:
        merged = pd.merge(
            world_bank_df,
            imf_df,
            on=["Country", "Year"],
            how="inner",
        )
        return merged

    @task(task_id="clean_and_impute")
    def clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
        return EconomicImputer().fit_transform(df)

    @task(task_id="extract_features")
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        return Feature_Extraction().create_features(df)

    @task(task_id="split_dataset")
    def split_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {
            "train": df[df["Year"] <= 2012].copy(),
            "val": df[(df["Year"] > 2012) & (df["Year"] <= 2018)].copy(),
            "test": df[df["Year"] > 2018].copy(),
        }

    @task(task_id="select_features")
    def select_features(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        train = splits["train"]
        val = splits["val"]
        test = splits["test"]

        selector = Feature_Selection()
        feature_columns = [c for c in train.columns if c not in ["Country", "GDP"]]
        selected_vars = selector.variance(train, feature_columns, threshold=0.01)
        selected_corr, _ = selector.correlation(train, selected_vars, target="GDP", threshold=0.9)

        selected_cols = [c for c in selected_vars if c in selected_corr] + ["Country", "GDP"]
        return {
            "train": train[selected_cols].copy(),
            "val": val[selected_cols].copy(),
            "test": test[selected_cols].copy(),
        }

    @task(task_id="scale_datasets")
    def scale_datasets(splits: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        transformer = Make_Transform(ignore_cols=["Country"])
        return transformer.standard_scaler(splits["train"], splits["val"], splits["test"])

    @task(task_id="prepare_model_data")
    def prepare_model_data(splits: dict[str, pd.DataFrame]) -> dict:
        train = splits["train"]
        val = splits["val"]
        test = splits["test"]

        handler = Handling()
        train_, val_, test_, num_countries = handler.handle_country(train, val, test)

        train_ = train_.sort_values(["country_id", "Year"])
        val_ = val_.sort_values(["country_id", "Year"])
        test_ = test_.sort_values(["country_id", "Year"])

        target_col = "GDP"
        selected_features = [
            c for c in train_.columns if c not in [target_col, "country_id", "Year"]
        ]

        train_ds = handler.prepare_multi_output_ds_panel(train_, selected_features, target_col, window_size=5, batch_size=32)
        val_ds = handler.prepare_multi_output_ds_panel(val_, selected_features, target_col, window_size=5, batch_size=32)
        test_ds = handler.prepare_multi_output_ds_panel(test_, selected_features, target_col, window_size=5, batch_size=32)

        return {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
            "num_countries": num_countries,
            "num_features": len(selected_features),
        }

    @task(task_id="train_model")
    def train_model(model_data: dict) -> dict:
        model = Modeling.build_multi_target_lstm(
            num_countries=model_data["num_countries"],
            num_features=model_data["num_features"],
            num_targets=1,
            window_size=5,
        )

        model.fit(
            model_data["train"],
            validation_data=model_data["val"],
            epochs=10,
            verbose=1,
            callbacks=[
                __import__("tensorflow").keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    restore_best_weights=True,
                )
            ],
        )

        loss, mae, r2 = model.evaluate(model_data["test"], verbose=0)
        return {"loss": float(loss), "mae": float(mae), "r2": float(r2)}

    @task(task_id="report_metrics")
    def report_metrics(metrics: dict) -> str:
        return (
            f"trained model metrics: loss={metrics['loss']:.4f}, "
            f"mae={metrics['mae']:.4f}, r2={metrics['r2']:.4f}"
        )

    world_bank_df = scrape_world_bank()
    imf_df = scrape_exchange_rate()
    merged_df = merge_sources(world_bank_df, imf_df)
    cleaned_df = clean_and_impute(merged_df)
    features_df = extract_features(cleaned_df)
    splits = split_dataset(features_df)
    selected_splits = select_features(splits)
    scaled_splits = scale_datasets(selected_splits)
    model_data = prepare_model_data(scaled_splits)
    metrics = train_model(model_data)
    report_metrics(metrics)


def DAG():
    return macro_data_pipeline()
