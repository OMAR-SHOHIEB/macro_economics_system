from tensorflow.keras import layers, Model
import tensorflow as tf

class LSTMModeling:
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
