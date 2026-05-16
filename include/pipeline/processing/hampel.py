import numpy as np
import pandas as pd

class HampelFilter:
    """
    A robust outlier detection filter using the median and median absolute deviation (MAD).
    """

    def __init__(self, window_size=5, n_sigmas=3, country_col="Country"):
        self.window_size = window_size
        self.n_sigmas = n_sigmas
        self.country_col = country_col

    def _hampel_filter_series(self, x):
        """
        Applies the Hampel filter to a single pandas Series.
        """
        # Constant for normal distribution
        L = 1.4826
        
        # Calculate rolling median and MAD
        rolling_median = x.rolling(self.window_size, min_periods=1, center=True).median()
        rolling_mad = x.rolling(self.window_size, min_periods=1, center=True).apply(
            lambda v: np.median(np.abs(v - np.median(v))), raw=True
        )
        
        # Identify outliers
        threshold = self.n_sigmas * L * rolling_mad
        outliers = np.abs(x - rolling_median) > threshold
        
        # Replace outliers with rolling median
        x_cleaned = x.copy()
        x_cleaned[outliers] = rolling_median[outliers]
        
        return x_cleaned

    def transform(self, df, columns=None):
        """
        Applies the Hampel filter to the specified columns of the dataframe, 
        grouped by the country column.
        """
        df = df.copy()
        
        if columns is None:
            # Detect all numeric columns except the country and any typical time columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.country_col in columns:
                columns.remove(self.country_col)
            if 'Year' in columns:
                columns.remove('Year')

        for col in columns:
            df[col] = df.groupby(self.country_col)[col].transform(self._hampel_filter_series)
            
        return df
