import numpy as np

from sklearn.preprocessing import (
    MinMaxScaler,
    LabelEncoder
)

from data.feature_engineering import (
    engineer_features
)


def preprocess_data(df, target_column):

    label_enc = LabelEncoder()

    df["Country_id"] = label_enc.fit_transform(
        df["Country"]
    )

    df = df.set_index("Country")

    df = (

        df
        .groupby("Country", group_keys=False)
        .apply(engineer_features)
    )

    df = df.reset_index()

    numeric_cols = (

        df
        .select_dtypes(include=[np.number])
        .columns
        .tolist()
    )

    for col in numeric_cols:

        df[col] = (

            df
            .groupby("Country")[col]
            .transform(
                lambda s:
                s.interpolate("linear")
                .bfill()
                .ffill()
            )
        )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df[numeric_cols] = (
        df[numeric_cols]
        .fillna(df[numeric_cols].median())
    )

    df[target_column] = np.log1p(
        df[target_column].clip(lower=0)
    )

    exclude = {"Country", target_column}

    feature_cols = [

        c for c in df.columns

        if c not in exclude
        and df[c].dtype != object
    ]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    return (
        df,
        feature_cols,
        scaler_X,
        scaler_y,
        label_enc
    )