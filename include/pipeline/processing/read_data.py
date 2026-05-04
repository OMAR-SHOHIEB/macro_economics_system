import pandas as pd
from pathlib import Path
from pathlib import Path


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
    
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
        self.RAW_PATH = self.BASE_DIR / "data" / "processed" / "data_after_clean_delete_null.csv"
        self.read_after_null()
    
    def read_after_null(self):
        return pd.read_csv(self.RAW_PATH)
    
        
class Read_Feature_Extracted():
    
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
        self.RAW_PATH = self.BASE_DIR / "data" / "processed" / "feature_extraction.csv"
        self.read_after_null()
    
    def read_after_null(self):
        def _check_files(self):
            path =self.RAW_PATH
            if not path.exists():
                    raise FileNotFoundError(f"{path} not found, please run Feature_Extracted.py file before run this file.")
        _check_files()
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
                    raise FileNotFoundError(f"{path} not found.")
        _check_files(self)
        train = pd.read_csv(self.train_path)
        val   = pd.read_csv(self.val_path)
        test  = pd.read_csv(self.test_path)

        return train, val, test
        
        
class Read_Selected:
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parents[3]

        self.train_path = self.BASE_DIR / "data" / "processed" / 'scaled_transformed' / "train_.csv"
        self.val_path = self.BASE_DIR / "data" / "processed" / 'scaled_transformed' / "val_.csv"
        self.test_path = self.BASE_DIR / "data" / "processed" / 'scaled_transformed' / "test.csv"

    def load_data(self):
        def _check_files(self):
            for path in [self.train_path, self.val_path, self.test_path]:
                if not path.exists():
                    raise FileNotFoundError(f"{path} not found.")
        _check_files(self)
        train = pd.read_csv(self.train_path)
        val = pd.read_csv(self.val_path)
        test = pd.read_csv(self.test_path)

        return train, val, test


class Read_Merged:
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parents[3]
        self.merged_path = self.BASE_DIR / "data" / "processed" / "data_merged.csv"

    def load_data(self):
        if not self.merged_path.exists():
            raise FileNotFoundError(f"{self.merged_path} not found. Run merge_raw_sources first.")
        return pd.read_csv(self.merged_path)


class Read_SelectedSplit:
    def __init__(self):
        self.BASE_DIR = Path(__file__).resolve().parents[3]
        self.train_path = self.BASE_DIR / "data" / "processed" / "selected" / "train_data.csv"
        self.val_path = self.BASE_DIR / "data" / "processed" / "selected" / "validation_data.csv"
        self.test_path = self.BASE_DIR / "data" / "processed" / "selected" / "test_data.csv"

    def load_data(self):
        def _check_files(self):
            for path in [self.train_path, self.val_path, self.test_path]:
                if not path.exists():
                    raise FileNotFoundError(f"{path} not found. Run select_features first.")
        _check_files(self)
        train = pd.read_csv(self.train_path)
        val = pd.read_csv(self.val_path)
        test  = pd.read_csv(self.test_path)

        return train, val, test