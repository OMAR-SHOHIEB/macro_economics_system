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
        
        df = df.sort_values(['Country', 'Year'])

        df['Trade_Balance'] = df['Exports'] - df['Imports']

        df['Inflation_diff'] = df.groupby('Country')['Inflation Rate'].pct_change()
        df['Unemployment_diff'] = df.groupby('Country')['Unemployment Rate'].pct_change()
        df['GDP_growth_calc'] = df.groupby('Country')['GDP'].pct_change()
        df['Population_growth'] = df.groupby('Country')['Population'].pct_change()
        df['Investment_growth'] = df.groupby('Country')['Investment'].pct_change()

        # this is improtant because it give us the retio 
        df['Investment_to_GDP'] = df['Investment'] / df['GDP']
        df['Trade_to_GDP'] = df['Trade_Balance'] / df['GDP']

        df['Inflation_volatility'] = df.groupby('Country')['Inflation Rate'].shift(1).rolling(5).std()

        # The inflation in this year >? inflation to the last 5 years + upper threshold ----> detect the shocks that crisis make or other things
        df['Inflation_shock'] = (df['Inflation Rate'] > df['Inflation Rate'].rolling(5).mean() + 2*df['Inflation Rate'].rolling(5).std()).astype(int)
        return self.Final_data(df)
    
    def Final_data(self, df):
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()    
        return df






if __name__ == "__main__":
    data = Read_AfterNull()
    data=data.read_after_null()
    print(data.head())
    
    feature=Feature_Extraction()
    df=feature.create_features(data)
    feature_extraction(df)
    save=Make_Splitting()
    save.splitting(df)
    