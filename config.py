
import os 

TARGET_COLUMN = "GDP"

LOOKBACK = 10

FORECAST_HORIZON = 1

TEST_SIZE = 0.2

EMBEDDING_DIM = 8

USE_COUNTRY_EMBED = True

EPOCHS = 150

BATCH_SIZE = 64
DATA_PATH= os.path.join (os.getcwd(), "data_world_bank.csv")