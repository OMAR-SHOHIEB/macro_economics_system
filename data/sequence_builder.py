import numpy as np


def create_sequences(
    df,
    feature_cols,
    target_column,
    lookback,
    forecast_horizon,
    test_size
):

    X_tr, y_tr, c_tr = [], [], []
    X_te, y_te, c_te = [], [], []

    all_years = sorted(df["Year"].unique())

    cutoff_idx = int(
        len(all_years) * (1 - test_size)
    )

    cutoff_year = all_years[cutoff_idx]

    for country, grp in df.groupby("Country"):

        grp = (
            grp
            .sort_values("Year")
            .reset_index(drop=True)
        )

        X_arr = grp[feature_cols].values
        y_arr = grp[target_column].values
        yr_arr = grp["Year"].values

        cid = grp["Country_id"].iloc[0]

        for i in range(
            lookback,
            len(X_arr) - forecast_horizon
        ):

            target_year = (
                yr_arr[i + forecast_horizon]
            )

            seq_X = X_arr[
                i - lookback:i
            ]

            seq_y = y_arr[
                i + forecast_horizon
            ]

            if target_year < cutoff_year:

                X_tr.append(seq_X)
                y_tr.append(seq_y)
                c_tr.append(cid)

            else:

                X_te.append(seq_X)
                y_te.append(seq_y)
                c_te.append(cid)

    return (
        np.array(X_tr),
        np.array(y_tr),
        np.array(c_tr),
        np.array(X_te),
        np.array(y_te),
        np.array(c_te),
        cutoff_year
    )