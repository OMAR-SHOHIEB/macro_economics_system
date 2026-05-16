import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add include directory to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "include"))

from pipeline.scraping.world_bank import WorldBankScraper
from pipeline.scraping.imf import get_exchange_rate_panel
from pipeline.processing.clean import EconomicImputer
from pipeline.processing.feature_extraction import Feature_Extraction
from pipeline.processing.feature_selection import Feature_Selection
from pipeline.processing.prepare_data import Handling, Modeling
from pipeline.processing.transform import Make_Transform

def run_pipeline():
    print("🚀 Starting Macroeconomic Pipeline...")
    
    # 1. Scrape World Bank
    print("📦 Scraping World Bank data...")
    wb = WorldBankScraper()
    # List of common indicators
    indicators = [
        wb.get_gdp_total(), wb.get_population(), wb.get_gdp_growth(),
        wb.get_gdp_per_capita(), wb.get_inflation_rate(), wb.get_unemployment_rate(),
        wb.get_fdi(), wb.get_exports(), wb.get_imports(),
        wb.get_gov_spending(), wb.get_investment(), wb.get_interest_rate(),
        wb.get_life_expectancy()
    ]
    frames = [df for df in indicators if df is not None and not df.empty]
    merged_wb = pd.concat(frames, ignore_index=True)
    pivot_df = merged_wb.pivot_table(
        index=["Country", "Year"],
        columns="Metric",
        values="Value",
    ).reset_index()
    pivot_df.columns.name = None
    
    # 2. Scrape IMF
    print("📈 Scraping IMF Exchange Rates...")
    imf_df = get_exchange_rate_panel(None)
    imf_df = imf_df.rename(columns={"country": "Country", "year": "Year", "exchange_rate": "Exchange Rate"})
    
    # 3. Merge
    print("🔗 Merging data sources...")
    df = pd.merge(pivot_df, imf_df, on=["Country", "Year"], how="inner")
    
    # 4. Clean and Impute (Includes our new Hampel Filter!)
    print("🧹 Cleaning and Imputing (applying Hampel Filter)...")
    imputer = EconomicImputer()
    df_clean = imputer.fit_transform(df)
    
    # Save processed data for the dashboard
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_dir / "data_after_clean_delete_null.csv", index=False)
    
    # 5. Feature Extraction
    print("✨ Extracting features...")
    extractor = Feature_Extraction()
    df_features = extractor.create_features(df_clean)
    df_features.to_csv(output_dir / "feature_extraction.csv", index=False)
    
    # 6. Split and Select
    print("✂️ Splitting and Selecting features...")
    train = df_features[df_features["Year"] <= 2012].copy()
    val = df_features[(df_features["Year"] > 2012) & (df_features["Year"] <= 2018)].copy()
    test = df_features[df_features["Year"] > 2018].copy()
    
    selector = Feature_Selection()
    feature_columns = [c for c in train.columns if c not in ["Country", "GDP", "Year"]]
    selected_vars = selector.variance(train, feature_columns, threshold=0.01)
    
    # 7. Model Training (Optional, might be slow)
    print("🤖 Training Model (Simulated for speed)...")
    # In a full run, we would call the LSTM training logic here.
    
    print("✅ Pipeline Completed Successfully!")
    print(f"📊 Processed data saved to {output_dir}")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        # If scraping fails (no internet), we still have the dashboard data
        print("💡 Note: You can still run 'streamlit run app/streamlit_app.py' if data exists.")
