from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Concatenate,
    LayerNormalization
)

from tensorflow.keras.optimizers import Adam

from models.attention import attention_block


def build_bilstm_model(
    lookback,
    n_features,
    n_countries,
    embedding_dim,
    use_country_embed=True
):

    seq_input = Input(
        shape=(lookback, n_features)
    )

    x = Bidirectional(
        LSTM(
            64,
            return_sequences=True
        )
    )(seq_input)

    x = Dropout(0.3)(x)

    x = attention_block(
        x,
        num_heads=2,
        key_dim=16
    )

    x = Bidirectional(
        LSTM(
            32,
            return_sequences=False
        )
    )(x)

    x = Dropout(0.2)(x)

    if use_country_embed:

        country_input = Input(shape=(1,))

        emb = Embedding(
            input_dim=n_countries,
            output_dim=embedding_dim
        )(country_input)

        emb = Flatten()(emb)

        x = Concatenate()([x, emb])

        inputs = [seq_input, country_input]

    else:

        inputs = seq_input

    x = Dense(64, activation="relu")(x)

    x = LayerNormalization()(x)

    x = Dense(32, activation="relu")(x)

    output = Dense(1)(x)

    model = Model(
        inputs=inputs,
        outputs=output
    )

    model.compile(

        optimizer=Adam(
            learning_rate=3e-4,
            clipnorm=1.0
        ),

        loss="huber",
        metrics=["mae"]
    )

    return model