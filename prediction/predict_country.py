import numpy as np


def predict_country(predictor, country_name, model_type="bilstm"):

    grp = predictor.df[predictor.df["Country"] == country_name].sort_values("Year")

    grp = grp.tail(predictor.lookback)

    X = predictor.scaler_X.transform(grp[predictor.feature_cols].values)
    X = X.reshape(1, predictor.lookback, -1)

    c_id = np.array([[grp["Country_id"].iloc[0]]])

    if model_type == "bilstm":
        pred = predictor.model.predict([X, c_id], verbose=0)

    elif model_type == "xgboost":
        pred = predictor.xgb_model.predict(X.reshape(1, -1)).reshape(-1, 1)

    else:
        pred = predictor.lgbm_model.predict(X.reshape(1, -1)).reshape(-1, 1)

    pred = np.expm1(predictor.scaler_y.inverse_transform(pred))[0][0]

    year = grp["Year"].max() + predictor.forecast_horizon

    print(country_name, "|", year, "|", model_type, "|", f"${pred:,.0f}")

    return pred