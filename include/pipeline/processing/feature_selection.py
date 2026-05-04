from read_data import ReadTrainValTest
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from transform import Make_Transform
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from pipeline.storage.save_csv import transfrom_select_save


class Feature_Selection:
    def __init__(self):
        self.ignore_cols = ['Country']
        self.deleted_var = set()
    def variance(self, train, features, threshold=0.01):

        features = [f for f in features if f not in self.ignore_cols]

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(train[features])

        selected = train[features].columns[selector.get_support()]

        self.deleted_var = set(features) - set(selected)

        return list(selected)
    
    def transform(self, train, val, test):
        trs=Make_Transform()
        # train, val, test=trs.log1p(train, val, test)
        train, val, test=trs.standard_scaler(train, val, test)
        return train, val, test
        
    def correlation(self, train, features, target, threshold=0.9):

        features = [f for f in features if f not in self.ignore_cols]

        df = train[features + [target]].copy()

        corr_matrix = df.corr(numeric_only=True)


        upper = corr_matrix.abs().where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            col for col in upper.columns
            if any(upper[col] > threshold)
        ]

        selected = list(set(features) - set(to_drop))

        return selected, to_drop


    def model_based(self, train, features, target, th=0.001):

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

        features = [f for f in features if f not in self.ignore_cols]

        model.fit(train[features], train[target])

        importance = pd.Series(model.feature_importances_, index=features)

        selected = importance[importance > th].index.tolist()
        dropped  = importance[importance <= th].index.tolist()

        return selected, dropped

if __name__ == "__main__":

    data = ReadTrainValTest()
    train_, val_, test_ = data.load_data()

    print("Train shape:", train_.shape)

    features = list(train_.columns)
    target = "GDP"  

    fs = Feature_Selection()

    selected_var = fs.variance(train_, features)
    # make transformation
    train, val, test = fs.transform(train_[selected_var].copy(), val_[selected_var].copy(), test_[selected_var].copy())
    

    selected_corr, dropped_corr = fs.correlation(train, selected_var, target)

    selected_model = fs.model_based(train, selected_corr, target)
    print(selected_model)
    
    train_final = train_[selected_corr + ['Country', "GDP"]]
    val_final = val_[selected_corr + ['Country', "GDP"]]
    test_final = test_[selected_corr + ['Country', "GDP"]]
    # save
    transfrom_select_save(train_final, val_final, test_final)
    
    
