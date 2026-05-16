from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import numpy as np
from read_data import Read_Selected


class Handling():
    def __init__(self):
        self.le = LabelEncoder()

    def handle_country(self, train_, test_):
        train_ = train_.copy()
        test_  = test_.copy()

        train_['country_id'] = self.le.fit_transform(train_['Country'])
        test_['country_id']  = self.le.transform(test_['Country'])

        train_ = train_.drop(columns=['Country'])
        test_  = test_.drop(columns=['Country'])

        num_countries = len(self.le.classes_)

        return train_, test_, num_countries

    @staticmethod
    def prepare_multi_output_ds_panel(df, feature_cols, target_cols, window_size=5, batch_size=32):
        X_country = []
        X_ts      = []
        y_target  = []

        print(f"Preparing panel data with {len(df['country_id'].unique())} countries...")

        for country_id, group in df.groupby('country_id', sort=True):
            group    = group.sort_values('Year').reset_index(drop=True)
            n_periods = len(group)

            if n_periods < window_size:
                print(f"  Warning: Country {country_id} has {n_periods} periods < window_size={window_size}, skipping")
                continue

            ts_data     = group[feature_cols].values.astype('float32')
            target_data = group[target_cols].values.astype('float32').flatten()
            n_windows   = n_periods - window_size

            for i in range(n_windows):
                X_country.append(country_id)
                X_ts.append(ts_data[i:i+window_size])
                y_target.append(target_data[i+window_size])

            print(f"  Country {country_id}: {n_windows} windows created")

        X_country = np.array(X_country, dtype=np.int32)
        X_ts      = np.array(X_ts,      dtype=np.float32)
        y_target  = np.array(y_target,  dtype=np.float32)

        print(f"\nTotal windows : {len(X_country)}")
        print(f"X_ts shape    : {X_ts.shape}")
        print(f"y_target shape: {y_target.shape}\n")

        ds = tf.data.Dataset.from_tensor_slices((X_country, X_ts, y_target))

        def map_fn(country, ts, target):
            return {"country_in": country, "ts_in": ts}, target

        ds = ds.map(map_fn)
        return ds.shuffle(min(1000, len(X_country))).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def predict_by_country(self, model, df_full, country_name, feature_cols, target_col, window_size=5):
        if country_name not in self.le.classes_:
            print(f"Country '{country_name}' not found. Available countries:")
            print(list(self.le.classes_))
            return None

        country_id = self.le.transform([country_name])[0]
        country_df = df_full[df_full['country_id'] == country_id].sort_values('Year')

        if len(country_df) < window_size:
            print(f"Not enough data for '{country_name}'. Required: {window_size}, available: {len(country_df)}")
            return None

        last_window   = country_df[feature_cols].values[-window_size:].astype('float32')
        ts_input      = last_window[np.newaxis, :, :]
        country_input = np.array([[country_id]], dtype=np.int32)

        prediction      = model.predict({"country_in": country_input, "ts_in": ts_input}, verbose=0)
        predicted_value = prediction[0][0]
        last_year       = country_df['Year'].max()

        print(f"\nCountry         : {country_name}")
        print(f"Last year in data: {int(last_year)}")
        print(f"Predicted GDP    : {predicted_value:.4f}")

        return predicted_value


class R2ScoreExact(tf.keras.metrics.Metric):
    def __init__(self, name='r2_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.ss_res = self.add_weight(name='ss_res', initializer='zeros')
        self.sum_y  = self.add_weight(name='sum_y',  initializer='zeros')
        self.sum_y2 = self.add_weight(name='sum_y2', initializer='zeros')
        self.count  = self.add_weight(name='count',  initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        self.ss_res.assign_add(tf.reduce_sum(tf.square(y_true - y_pred)))
        self.sum_y.assign_add(tf.reduce_sum(y_true))
        self.sum_y2.assign_add(tf.reduce_sum(tf.square(y_true)))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        mean_y = self.sum_y / (self.count + tf.keras.backend.epsilon())
        ss_tot = self.sum_y2 - self.count * tf.square(mean_y)
        return 1.0 - self.ss_res / (ss_tot + tf.keras.backend.epsilon())

    def reset_state(self):
        for w in [self.ss_res, self.sum_y, self.sum_y2, self.count]:
            w.assign(0.)


class Modeling():
    @staticmethod
    def build_multi_target_lstm(num_countries, num_features, num_targets, window_size=5):
        country_in = layers.Input(shape=(1,), name="country_in", dtype=tf.int32)
        emb        = layers.Embedding(num_countries, 8)(country_in)
        emb        = layers.Reshape((8,))(emb)
        emb_seq    = layers.RepeatVector(window_size)(emb)

        ts_in  = layers.Input(shape=(window_size, num_features), name="ts_in")
        merged = layers.Concatenate()([ts_in, emb_seq])

        x = layers.LSTM(64, return_sequences=True, dropout=0.2)(merged)
        x = layers.LSTM(32, dropout=0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        output = layers.Dense(num_targets, name="output")(x)
        model  = Model(inputs=[country_in, ts_in], outputs=output)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='mse',
            metrics=['mae', R2ScoreExact()]
        )

        return model


if __name__ == "__main__":

    obj = Read_Selected()
    train_, test_ = obj.load_data()

    print("=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    print(f"Train shape: {train_.shape}")
    print(f"Test shape : {test_.shape}")
    print(f"\nData types:\n{train_.dtypes}\n")

    H = Handling()
    train_, test_, num_countries = H.handle_country(train_, test_)

    train_full = train_.copy()
    test_full  = test_.copy()

    print(f"Number of countries: {num_countries}")

    train_ = train_.sort_values(['country_id', 'Year'])
    test_  = test_.sort_values(['country_id', 'Year'])

    target_col = "GDP"

    selected_features = [c for c in train_.columns
                         if c not in (target_col, 'country_id', 'Year')]

    print(f"Selected features ({len(selected_features)}): {selected_features}")

    window = 5

    print("\n" + "=" * 60)
    print("PREPARING DATASETS")
    print("=" * 60)

    train_ds = H.prepare_multi_output_ds_panel(train_, selected_features, target_col, window)
    test_ds  = H.prepare_multi_output_ds_panel(test_,  selected_features, target_col, window)

    print("\n" + "=" * 60)
    print("VERIFYING DATA SHAPES")
    print("=" * 60)

    for batch_x, batch_y in train_ds.take(1):
        print(f"Country input: {batch_x['country_in'].shape}")
        print(f"TS input     : {batch_x['ts_in'].shape}")
        print(f"Target shape : {batch_y.shape}")

    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)

    md    = Modeling()
    model = md.build_multi_target_lstm(
        num_countries=num_countries,
        num_features=len(selected_features),
        num_targets=1,
        window_size=window
    )
    print(model.summary())

    model.fit(
        train_ds,
        epochs=50,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )

    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)

    test_loss, test_mae, test_r2 = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Test MAE       : {test_mae:.6f}")
    print(f"Test R2        : {test_r2:.6f}")

    print("\n" + "=" * 60)
    print("COUNTRY PREDICTION")
    print("=" * 60)

    country_name = input("Enter country name: ").strip()

    H.predict_by_country(
        model        = model,
        df_full      = test_full,
        country_name = country_name,
        feature_cols = selected_features,
        target_col   = target_col,
        window_size  = window
    )