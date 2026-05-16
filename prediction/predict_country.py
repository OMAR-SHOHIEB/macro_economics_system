import numpy as np


def predict_country(pipeline, country_name, model_type="bilstm"):

    # ─────────────────────────────
    # Get country data
    # ─────────────────────────────
    grp = pipeline.df[
        pipeline.df["Country"] == country_name
    ].sort_values("Year")

    # ─────────────────────────────
    # Safety checks
    # ─────────────────────────────
    if grp.empty:
        raise ValueError(f"Country not found: {country_name}")

    if len(grp) < pipeline.lookback:
        raise ValueError(
            f"Not enough data for {country_name}. "
            f"Need {pipeline.lookback}, got {len(grp)}"
        )

    # ─────────────────────────────
    # Take last sequence
    # ─────────────────────────────
    grp = grp.tail(pipeline.lookback)

    X_raw = grp[pipeline.feature_cols].values

    # IMPORTANT: scaler must NOT receive empty data
    if X_raw.shape[0] == 0:
        raise ValueError("Empty feature array after selection")

    X = pipeline.scaler_X.transform(X_raw)

    X = X.reshape(1, pipeline.lookback, -1)

    c_id = np.array([[grp["Country_id"].iloc[0]]])

    # ─────────────────────────────
    # Model selection
    # ─────────────────────────────
    if model_type == "bilstm":

        pred = pipeline.model.predict([X, c_id], verbose=0)

    elif model_type == "xgboost":

        pred = pipeline.xgb_model.predict(
            X.reshape(1, -1)
        ).reshape(-1, 1)

    elif model_type == "lightgbm":

        pred = pipeline.lgbm_model.predict(
            X.reshape(1, -1)
        ).reshape(-1, 1)

    else:

        raise ValueError(
            "model_type must be: bilstm, xgboost, lightgbm"
        )

    # ─────────────────────────────
    # Inverse transform
    # ─────────────────────────────
    pred = np.expm1(
        pipeline.scaler_y.inverse_transform(pred)
    )[0][0]

    # ─────────────────────────────
    # Forecast year
    # ─────────────────────────────
    year = grp["Year"].max() + pipeline.forecast_horizon

    print(
        f"{country_name} | {year} | {model_type} | ${pred:,.0f}"
    )

    return {
        "country": country_name,
        "year": year,
        "model": model_type,
        "gdp": float(pred)
    }