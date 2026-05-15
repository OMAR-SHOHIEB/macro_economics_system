import pandas as pd


def load_data(file_path):

    df = pd.read_csv(file_path)

    df = (

        df
        .sort_values(["Country", "Year"])
        .reset_index(drop=True)
    )

    return df