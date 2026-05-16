from read_data import Read_AfterNull
from Splitting import Make_Splitting
import pandas as pd
import numpy as np 
from  pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from pipeline.storage.save_csv import feature_extraction

class Feature_Extraction():
    def __init__(self):
        pass 
    

    def create_features(self, df):
        """
        Creates new economic features based on the provided dataframe.
        """
        df = df.copy()
        df = df.sort_values(['Country', 'Year'])

        # 1. Trade Balance
        df['Trade_Balance'] = df['Exports'] - df['Imports']

        # 2. Global Average Features (Averaged by Year across all countries)
        global_stats = df.groupby('Year')[['GDP Growth', 'Inflation Rate']].mean().reset_index()
        global_stats.rename(columns={
            'GDP Growth': 'Global_Avg_GDP_Growth', 
            'Inflation Rate': 'Global_Avg_Inflation'
        }, inplace=True)
        df = df.merge(global_stats, on='Year', how='left')

        # 3. Crisis Years
        crisis_years = [
            1980, 1981, 1982,  # Early 1980s Global Recession
            1990, 1991,        # Early 1990s Recession
            1997, 1998,        # Asian Financial Crisis
            2001,              # Dot-com Bubble Burst & 9/11 Shock
            2008, 2009,        # Great Financial Crisis
            2020               # COVID-19 Global Recession
        ]
        df['Is_Crisis_Year'] = df['Year'].isin(crisis_years).astype(int)

        # 4. Rolling Volatility (Last 5 years, shifted by 1 to avoid leakage)
        df['Inflation_volatility'] = df.groupby('Country')['Inflation Rate'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).std()
        )
        df['GDP_volatility'] = df.groupby('Country')['GDP Growth'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).std()
        )

        # 5. Inflation Shock
        def detect_shock(x):
            mean = x.rolling(5, min_periods=1).mean()
            std = x.rolling(5, min_periods=1).std()
            return (x > mean + 2 * std).astype(int)
        
        df['Inflation_shock'] = df.groupby('Country')['Inflation Rate'].transform(detect_shock)

        # 6. Additional Calculated Growths and Ratios
        df['Inflation_diff'] = df.groupby('Country')['Inflation Rate'].pct_change()
        df['Unemployment_diff'] = df.groupby('Country')['Unemployment Rate'].pct_change()
        df['GDP_growth_calc'] = df.groupby('Country')['GDP'].pct_change()
        df['Population_growth'] = df.groupby('Country')['Population'].pct_change()
        df['Investment_growth'] = df.groupby('Country')['Investment'].pct_change()

        df['Investment_to_GDP'] = df['Investment'] / df['GDP']
        df['Trade_to_GDP'] = df['Trade_Balance'] / df['GDP']

        # Fill NaNs created by rolling/diff operations
        df.fillna(0, inplace=True)
        
        return self.Final_data(df)






if __name__ == "__main__":
    data = Read_AfterNull()
    data=data.read_after_null()
    print(data.head())
    
    feature=Feature_Extraction()
    df=feature.create_features(data)
    feature_extraction(df)
    save=Make_Splitting()
    save.splitting(df)
    