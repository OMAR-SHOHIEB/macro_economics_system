from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from pipeline.storage.save_csv import splitting_save


class Make_Splitting():
    def  __init__(self):
        pass
    
    def splitting(self, df):
        train = df[df['Year'] <= 2018]
        test  = df[df['Year'] > 2018]
        splitting_save(train, test)