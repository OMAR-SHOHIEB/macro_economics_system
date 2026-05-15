import numpy as np


def engineer_features(grp):

    g = grp.copy()

    if "Exports" in g.columns and "Imports" in g.columns:
        g["Trade_Balance"] = (
            g["Exports"] - g["Imports"]
        )

    for lag in [1, 2, 3]:
        g[f"GDP_lag_{lag}"] = (
            g["GDP"].shift(lag)
        )

    g["GDP_roll3_mean"] = (
        g["GDP"].rolling(3).mean()
    )

    g["GDP_roll3_std"] = (
        g["GDP"].rolling(3).std()
    )

    g["GDP_YoY_pct"] = (
        g["GDP"].pct_change()
    )

    if "Population" in g.columns:
        g["Pop_growth"] = (
            g["Population"].pct_change()
        )

    return g