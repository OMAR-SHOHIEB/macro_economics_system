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
            return {"country_in": country, "ts_in": ts}, target
        
        ds = ds.map(map_fn)
        
        # Shuffle, batch, and prefetch
        return ds.shuffle(min(1000, len(X_country))).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    
class Modeling():
    @staticmethod
    def build_multi_target_lstm(num_countries, num_features, num_targets, window_size=5):
        """
        Build LSTM model for panel data time series forecasting.
        
        Args:
            num_countries: Number of unique countries
            num_features: Number of features in each timestep
            num_targets: Number of output targets (usually 1)
            window_size: Size of input windows
            
        Returns:
            Compiled Keras model
        """
        # Country embedding branch
        country_in = layers.Input(shape=(1,), name="country_in", dtype=tf.int32)
        emb = layers.Embedding(num_countries, 8)(country_in)
        emb = layers.Reshape((8,))(emb)
        emb_seq = layers.RepeatVector(window_size)(emb)  # (batch, window_size, 8)

        # Time series input branch
        ts_in = layers.Input(shape=(window_size, num_features), name="ts_in")

        # Merge country embedding with time series features
        merged = layers.Concatenate()([ts_in, emb_seq])  # (batch, window_size, num_features+8)

        # LSTM layers
        x = layers.LSTM(64, return_sequences=True, dropout=0.2)(merged)
        x = layers.LSTM(32, dropout=0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        # Output layer
        output = layers.Dense(num_targets, name="output")(x)

        model = Model(inputs=[country_in, ts_in], outputs=output)

        # Custom R² metric
        def r2_score(y_true, y_pred):
            ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
            ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
            return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', r2_score]
        )

        return model


if __name__ == "__main__":

    # Load data
    obj = Read_Selected()
    train_, val_, test_ = obj.load_data()

    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    print(f"Train shape: {train_.shape}")
    print(f"Val shape: {val_.shape}")
    print(f"Test shape: {test_.shape}")
    print(f"\nData types:\n{train_.dtypes}\n")

    # Handle country encoding
    H = Handling()
    train_, val_, test_, num_countries = H.handle_country(train_, val_, test_)
    
    print(f"Number of countries: {num_countries}")

    # Sort time series WITHIN each country (CRITICAL for panel data)
    train_ = train_.sort_values(['country_id', 'Year'])
    val_   = val_.sort_values(['country_id', 'Year'])
    test_  = test_.sort_values(['country_id', 'Year'])

    target_col = "GDP"

    # Remove target and metadata columns from features
    selected_features = [c for c in train_.columns 
                        if c != target_col and c != 'country_id' and c != 'Year']
    
    print(f"Selected features: {selected_features}")

    window = 5

    print("\n" + "=" * 60)
    print("PREPARING DATASETS")
    print("=" * 60)
    
    # Prepare datasets with proper panel data handling
    train = H.prepare_multi_output_ds_panel(train_, selected_features, target_col, window)
    val     = H.prepare_multi_output_ds_panel(val_, selected_features, target_col, window)
    test   = H.prepare_multi_output_ds_panel(test_, selected_features, target_col, window)

    # Verify shapes
    print("\n" + "=" * 60)
    print("VERIFYING DATA SHAPES")
    print("=" * 60)
    
    for batch_x, batch_y in train.take(1):
        print(f"Country input shape: {batch_x['country_in'].shape} (batch,)")
        print(f"TS input shape: {batch_x['ts_in'].shape} (batch, window_size, n_features)")
        print(f"Target shape: {batch_y.shape} (batch,)")
        print(f"\nSample country ID: {batch_x['country_in'][0].numpy()}")
        print(f"Sample target value: {batch_y[0].numpy():.2f}")

    # Build model
    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)
    
    md = Modeling()

    model = md.build_multi_target_lstm(
        num_countries=num_countries,
        num_features=len(selected_features),
        num_targets=1,
        window_size=window
    )

    print(model.summary())
    
    model.fit(
        train,
        validation_data=val,
        epochs=50,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    test_loss, test_mae, test_r2 = model.evaluate(test, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test R²: {test_r2:.6f}")