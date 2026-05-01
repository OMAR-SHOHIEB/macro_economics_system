import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

import sys
sys.path.append(r"D:\project\Data Science\macro_data_project")
from include.pipeline.storage.save_csv import data_clean
from sklearn.model_selection import train_test_split



class ReadMerge:
    
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent.parent

        self.dfi = self.read_imf('data_imf.csv')
        self.dfw = self.read_world_bank('data_world_bank.csv')

    def read_world_bank(self, file_name):
        raw_path = self.BASE_DIR /"data" / "raw" / "world_bank"
        file_path = raw_path / file_name
        return pd.read_csv(file_path)

    def read_imf(self, file_name):
        raw_path = self.BASE_DIR  /"data" / "raw" / "imf"
        file_path = raw_path / file_name
        return pd.read_csv(file_path)

    def final(self):
        self.dfi.rename(columns={
        "country": "Country",
        "year": "Year"
        }, inplace=True)
        
        self.dfi = self.dfi.drop(columns=["country_code"])
        return self.dfw.merge(
            self.dfi,
            on=["Country", "Year"],
            how="inner"
        )
        


class make_spliting():
    def  __init__(self):
        pass
    
    def splitting(self,df):
        train = df[df['Year'] <= 2012]
        val   = df[(df['Year'] > 2012) & (df['Year'] <= 2018)]
        test  = df[df['Year'] > 2018]
        return train, val, test
         
    
    
    
if __name__ == "__main__":
    data = ReadMerge()
    df = data.final()
    
    splitting = make_spliting()
    train, val, test=  splitting.splitting(df)