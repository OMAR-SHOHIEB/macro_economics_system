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
            RAW_PATH = BASE_DIR / "data" / "raw" 
            RAW_PATH.mkdir(parents=True, exist_ok=True)
            file_path = RAW_PATH / "data_after_clean_delete_null.csv"
            df.to_csv(file_path, index=False)
            print(f"Saved to: {file_path}")