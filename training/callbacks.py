from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)


def get_callbacks():

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6
        )
    ]

    return callbacks