from config import *

from data.loader import load_data
from data.preprocessing import preprocess_data
from data.sequence_builder import create_sequences

from models.bilstm_model import build_bilstm_model
from models.xgboost_model import build_xgboost_model
from models.lightgbm_model import build_lightgbm_model

from training.callbacks import get_callbacks
from prediction.predict_country import predict_country


class GDPPipeline:

    def __init__(self):

        self.df = None
        self.feature_cols = None
        self.scaler_X = None
        self.scaler_y = None

        self.bilstm = None
        self.xgb = None
        self.lgbm = None

    # ─────────────────────────────
    # Load + preprocess
    # ─────────────────────────────
    def prepare_data(self):

        self.df = load_data(DATA_PATH)

        (
            self.df,
            self.feature_cols,
            self.scaler_X,
            self.scaler_y,
            _
        ) = preprocess_data(
            self.df,
            TARGET_COLUMN
        )

        (
            self.X_train,
            self.y_train,
            self.c_train,
            self.X_test,
            self.y_test,
            self.c_test,
            _
        ) = create_sequences(
            self.df,
            self.feature_cols,
            TARGET_COLUMN,
            LOOKBACK,
            FORECAST_HORIZON,
            TEST_SIZE
        )

    # ─────────────────────────────
    # Scaling
    # ─────────────────────────────
    def scale(self):

        _, seq_len, n_features = self.X_train.shape

        self.X_train = self.scaler_X.fit_transform(
            self.X_train.reshape(-1, n_features)
        ).reshape(-1, seq_len, n_features)

        self.X_test = self.scaler_X.transform(
            self.X_test.reshape(-1, n_features)
        ).reshape(-1, seq_len, n_features)

        self.y_train = self.scaler_y.fit_transform(
            self.y_train.reshape(-1, 1)
        )

        self.y_test = self.scaler_y.transform(
            self.y_test.reshape(-1, 1)
        )


    def train(self):

        # BiLSTM
        self.bilstm = build_bilstm_model(
            LOOKBACK,
            len(self.feature_cols),
            self.df["Country_id"].nunique(),
            EMBEDDING_DIM
        )

        self.bilstm.fit(
            [self.X_train, self.c_train],
            self.y_train,
            validation_data=(
                [self.X_test, self.c_test],
                self.y_test
            ),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=get_callbacks(),
            shuffle=False,
            verbose=1
        )

        # XGBoost
        self.xgb = build_xgboost_model()
        self.xgb.fit(
            self.X_train.reshape(self.X_train.shape[0], -1),
            self.y_train.flatten()
        )

        # LightGBM
        self.lgbm = build_lightgbm_model()
        self.lgbm.fit(
            self.X_train.reshape(self.X_train.shape[0], -1),
            self.y_train.flatten()
        )

    # ─────────────────────────────
    # Predict
    # ─────────────────────────────
    def predict(self, country, model="bilstm"):

        return predict_country(
            self,
            country,
            model
        )



if __name__ == "__main__":

    pipeline = GDPPipeline()

    pipeline.prepare_data()
    pipeline.scale()
    pipeline.train()
    pipeline.predict("Germany", "xgboost")
    pipeline.predict("China", "lightgbm")