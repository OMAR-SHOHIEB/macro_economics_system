from xgboost import XGBRegressor


def build_xgboost_model():

    model = XGBRegressor(

        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    return model