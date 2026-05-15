from lightgbm import LGBMRegressor


def build_lightgbm_model():

    model = LGBMRegressor(

        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    return model