from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
from read_data import Read_Selected


class Handling():
    def __init__(self):
        pass 
    
    def handle_country(self, train_, val_, test_):
        le = LabelEncoder()
        
        train_ = train_.copy()
        val_   = val_.copy()
        test_  = test_.copy()
        
        train_['country_id'] = le.fit_transform(train_['Country'])
        val_['country_id']   = le.transform(val_['Country'])
        test_['country_id']  = le.transform(test_['Country'])
        
        train_ = train_.drop(columns=['Country'])
        val_ = val_.drop(columns=['Country'])
        test_ = test_.drop(columns=['Country'])
        
        num_countries = len(le.classes_)
        
        return train_, val_, test_, num_countries
    
    @staticmethod
    def prepare_multi_output_ds_panel(df, feature_cols, target_cols, window_size=5, batch_size=32):
        """
        Properly prepare panel data for time series forecasting.
        
        Creates windows WITHIN each country group to prevent data leakage
        between cross-sectional units.
        
        Args:
            df: DataFrame with columns ['country_id', 'Year', ...features..., target_cols]
            feature_cols: List of feature column names
            target_cols: Target column name (string, not list)
            window_size: Number of timesteps in each window
            batch_size: Batch size for training
            
        Returns:
            tf.data.Dataset with structure:
            - Input: {"country_in": (batch_size,), "ts_in": (batch_size, window_size, n_features)}
            - Output: (batch_size,)
        """
        X_country = []
        X_ts = []
        y_target = []
        
        print(f"Preparing panel data with {len(df['country_id'].unique())} countries...")
        
        # Process each country separately to avoid leakage
        for country_id, group in df.groupby('country_id', sort=True):
            # Sort by time within each country
            group = group.sort_values('Year').reset_index(drop=True)
            
            n_periods = len(group)
            
            if n_periods < window_size:
                print(f"  Warning: Country {country_id} has {n_periods} periods < window_size={window_size}, skipping")
                continue
            
            # Extract feature values: shape (n_periods, n_features)
            ts_data = group[feature_cols].values.astype('float32')
            
            # Extract target values: shape (n_periods,)
            target_data = group[target_cols].values.astype('float32').flatten()
            
            # Create windows for this country
            # Window i contains features from time [t, t+1, ..., t+window_size-1]
            # Target is the next value at t+window_size
            n_windows = n_periods - window_size
            
            for i in range(n_windows):
                X_country.append(country_id)
                X_ts.append(ts_data[i:i+window_size])  # (window_size, n_features)
                y_target.append(target_data[i+window_size])  # Next target value
            
            print(f"  Country {country_id}: {n_windows} windows created")
        
        # Convert to numpy arrays
        X_country = np.array(X_country, dtype=np.int32)
        X_ts = np.array(X_ts, dtype=np.float32)  # (n_total_windows, window_size, n_features)
        y_target = np.array(y_target, dtype=np.float32)  # (n_total_windows,)
        
        print(f"\nTotal windows: {len(X_country)}")
        print(f"X_ts shape: {X_ts.shape}")
        print(f"y_target shape: {y_target.shape}\n")
        
        # Create tf.data.Dataset
        ds = tf.data.Dataset.from_tensor_slices((
            X_country,
            X_ts,
            y_target
        ))
        
        # Map to proper input format
        def map_fn(country, ts, target):
            # The country input needs to be (None, 1) but Dataset from tensor slices 
            # might give (batch,) if not careful. The model expects (batch, 1) 
            # because of Embedding input_length or just Input shape (1,).
            # Actually layers.Input(shape=(1,)) means it expects an array of size 1.
            return {"country_in": tf.expand_dims(country, -1), "ts_in": ts}, target
        
        ds = ds.map(map_fn)
        
        # Shuffle, batch, and prefetch
        return ds.shuffle(min(1000, len(X_country))).batch(batch_size).prefetch(tf.data.AUTOTUNE)