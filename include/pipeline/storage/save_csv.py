from pathlib import Path
import pandas as pd

def world_save(df):
        BASE_DIR = Path(__file__).resolve().parent.parent
        RAW_PATH = BASE_DIR / "data" / "raw" / "world_bank"
        RAW_PATH.mkdir(parents=True, exist_ok=True)
        file_path = RAW_PATH / "data_world_bank.csv"
        df.to_csv(file_path, index=False)
        print(f"Saved to: {file_path}")
        
def imf_save(df):
            BASE_DIR = Path(__file__).resolve().parent.parent
            RAW_PATH = BASE_DIR / "data" / "raw" / "imf"
            RAW_PATH.mkdir(parents=True, exist_ok=True)
            file_path = RAW_PATH / "data_imf.csv"
            df.to_csv(file_path, index=False)
            print(f"Saved to: {file_path}")
            
            
def data_clean(df):
            BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
            RAW_PATH = BASE_DIR / "data" / "processed" 
            RAW_PATH.mkdir(parents=True, exist_ok=True)
            file_path = RAW_PATH / "data_after_clean_delete_null.csv"
            df.to_csv(file_path, index=False)
            print(f"Saved to: {file_path}")
            

def splitting_save(train, test ):
            BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
            RAW_PATH = BASE_DIR / "data" / "processed" /"split"
            RAW_PATH.mkdir(parents=True, exist_ok=True)
            file_path_trian = RAW_PATH / "train_data.csv"
            file_path_test = RAW_PATH / "test_data.csv"
            train.to_csv(file_path_trian, index=False)
            test.to_csv(file_path_test, index=False)
            
            
def feature_extraction(df):
            BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
            RAW_PATH = BASE_DIR / "data" / "processed" 
            RAW_PATH.mkdir(parents=True, exist_ok=True)
            file_path = RAW_PATH / "feature_extraction.csv"
            df.to_csv(file_path, index=False)
            print(f"Saved to: {file_path}")
            
            
def transfrom_select_save(train, test ):
            BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
            RAW_PATH = BASE_DIR / "data" / "processed" /"scaled_transformed"
            RAW_PATH.mkdir(parents=True, exist_ok=True)
            file_path_trian = RAW_PATH / "train_.csv"
            file_path_test = RAW_PATH / "test_.csv"
            train.to_csv(file_path_trian, index=False)
            test.to_csv(file_path_test, index=False)