import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from include.pipeline.storage.save_csv import splitting

from sklearn.model_selection import train_test_split


class make_spliting():
    def  __init__(self):
        pass
    
    def splitting(self,df):
        train = df[df['Year'] <= 2012]
        val   = df[(df['Year'] > 2012) & (df['Year'] <= 2018)]
        test  = df[df['Year'] > 2018]
        return train, val, test
         
    
    
        
if __name__ == "__main__":
    data=ReadMerge()
    df = data.get_data_aftermerge_split()
    
    splitting = make_spliting()
    train, val, test=  splitting.splitting(df)
    