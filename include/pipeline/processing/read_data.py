import pandas as pd
from pathlib import Path
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from include.pipeline.storage.save_csv import imf_save

class ReadMergeScrape:
    
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

    def get_data_after_merge(self):
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
        
        
class Read_AfterNull():
    
    def __init(self):
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
        self.RAW_PATH = self.BASE_DIR / "data" / "processed" / "data_after_clean_delete_null.csv"
        self.read_after_null()
    
    def read_after_null(self):
        return pd.read_csv(self.RAW_PATH)
    
class ReadTrainValTest:
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parents[3]

        self.train_path = self.BASE_DIR / "data" / "processed" / 'split' / "train_data.csv"
        self.val_path   = self.BASE_DIR / "data" / "processed" / 'split'/ "validation_data.csv"
        self.test_path  = self.BASE_DIR / "data" / "processed" / 'split' / "test_data.csv"

    def load_data(self):
        
        def _check_files(self):
             for path in [self.train_path, self.val_path, self.test_path]:
                 if not path.exists():
                    raise FileNotFoundError(f"{path} not found, please run clean.py file before run this file.")
        train = pd.read_csv(self.train_path)
        val   = pd.read_csv(self.val_path)
        test  = pd.read_csv(self.test_path)

        return train, val, test
        
        