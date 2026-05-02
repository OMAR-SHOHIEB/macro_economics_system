import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from include.pipeline.storage.save_csv import data_clean

from pipeline.processing.read_data import ReadMergeScrape

class EconomicImputer:
    def __init__(self, country_col="Country", time_col="Year"):
        self.country_col = country_col
        self.time_col = time_col
        self.models = {}


    def basic_impute(self, df):
        df = df.copy()
        df = df.sort_values([self.country_col, self.time_col])

        exclude = [self.country_col, self.time_col]
        cols = [c for c in df.columns if c not in exclude]

        for col in cols:
            if df[col].dtype == "object":
                continue

            if col in ["GDP", "Exports", "Imports", "Investment",
                       "Government Spending", "Foreign Direct Investment",
                       "GDP per Capita"]:

                df[col] = df.groupby(self.country_col)[col].transform(
                    lambda x: x.interpolate(method="linear", limit_direction="both")
                )


            elif col in ["Inflation Rate", "Interest Rate", "Unemployment Rate"]:

                df[col] = df.groupby(self.country_col)[col].transform(
                    lambda x: x.ffill().bfill()
                )


            elif col in ["Life Expectancy"]:

                df[col] = df.groupby(self.country_col)[col].transform(
                    lambda x: x.interpolate(limit_direction="both")
                )

        return df


    def model_impute(self, df, target):
        df = df.copy()

        train = df[df[target].notna()]
        test = df[df[target].isna()]

        if test.empty:
            return df

        features = [c for c in df.columns
                    if c not in [target, self.country_col]]


        train_enc = pd.get_dummies(train, columns=[self.country_col,], drop_first=True)
        test_enc = pd.get_dummies(test, columns=[self.country_col], drop_first=True)


        train_enc, test_enc = train_enc.align(test_enc, join="left", axis=1, fill_value=0)

        X_train = train_enc[features]
        y_train = train_enc[target]
        X_test = test_enc[features]

        model = RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        df.loc[df[target].isna(), target] = model.predict(X_test)

        self.models[target] = model

        return df

  
    def fit_transform(self, df, model_targets=None):
        df = self.basic_impute(df)

        if model_targets is None:
            model_targets = ["Exports", "Imports",
                             "Government Spending", "Investment", "GDP Growth"]

        for col in model_targets:
            if col in df.columns:
                df = self.model_impute(df, col)

        return df
    
    
    

if __name__ == "__main__":
    data=ReadMergeScrape()
    df = data.get_data_after_merge()
    
    imputer = EconomicImputer()
    df_clean = imputer.fit_transform(df)
    data_clean(df_clean)
    