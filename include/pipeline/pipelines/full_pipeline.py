import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "include" / "pipeline" / "processing"))
sys.path.append(str(PROJECT_ROOT / "include" / "pipeline" / "modeling"))

from read_data import ReadMergeScrape
from clean import EconomicImputer
from feature_extraction import Feature_Extraction
from Splitting import Make_Splitting
from transform import Make_Transform
from feature_selection import Feature_Selection
from prepare_data import Handling
from lstm import LSTMModeling

def run_full_pipeline():
    print("Starting Full Macroeconomic Modeling Pipeline...")

    # 1. Read and Merge Data
    print("\n--- Step 1: Reading and Merging Data ---")
    reader = ReadMergeScrape()
    df_raw = reader.get_data_after_merge()
    print(f"Raw data shape: {df_raw.shape}")

    # 2. Imputation
    print("\n--- Step 2: Data Imputation ---")
    imputer = EconomicImputer()
    df_clean = imputer.fit_transform(df_raw)
    print(f"Cleaned data shape: {df_clean.shape}")

    # 3. Feature Extraction
    print("\n--- Step 3: Feature Extraction ---")
    extractor = Feature_Extraction()
    df_features = extractor.create_features(df_clean)
    print(f"Data shape after feature extraction: {df_features.shape}")

    # 4. Splitting
    print("\n--- Step 4: Data Splitting ---")
    splitter = Make_Splitting()
    train, val, test = splitter.splitting(df_features)
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

    # 5. Transformation (Log1p, Scaling, PCA)
    print("\n--- Step 5: Transformation ---")
    transformer = Make_Transform(ignore_cols=['Country', 'Year', 'GDP'])
    
    # Log transformation
    train, val, test = transformer.log1p(train, val, test)
    # Robust Scaling
    train, val, test = transformer.robust_scaler(train, val, test)
    # PCA for highly correlated clusters
    train, val, test = transformer.pca_cluster_transformation(train, val, test, correlation_threshold=0.90)
    
    print(f"Transformed train shape: {train.shape}")

    # 6. Feature Selection (Variance & RFE)
    print("\n--- Step 6: Feature Selection ---")
    selector = Feature_Selection()
    target = 'GDP'
    
    # Variance Threshold
    initial_features = [c for c in train.columns if c not in ['Country', 'Year', target]]
    selected_var = selector.variance(train, initial_features, threshold=0.1)
    print(f"Features after variance threshold: {len(selected_var)}")
    
    # RFE with XGBoost
    selected_final, dropped = selector.recursive_feature_elimination(train, selected_var, target, n_features_to_select=10)
    print(f"Final selected features: {selected_final}")

    # 7. Prepare Panel Data for LSTM
    print("\n--- Step 7: Preparing Panel Data for LSTM ---")
    handler = Handling()
    
    # Keep only selected features + metadata
    cols_to_keep = ['Country', 'Year', target] + selected_final
    train = train[cols_to_keep]
    val = val[cols_to_keep]
    test = test[cols_to_keep]
    
    # Country encoding
    train, val, test, num_countries = handler.handle_country(train, val, test)
    
    window_size = 5
    batch_size = 32
    
    # Create tf.data.Datasets
    train_ds = handler.prepare_multi_output_ds_panel(train, selected_final, target, window_size, batch_size)
    val_ds = handler.prepare_multi_output_ds_panel(val, selected_final, target, window_size, batch_size)
    test_ds = handler.prepare_multi_output_ds_panel(test, selected_final, target, window_size, batch_size)

    # 8. Model Building and Training
    print("\n--- Step 8: Model Building and Training ---")
    model_builder = LSTMModeling()
    num_features = len(selected_final)
    
    model = model_builder.build_multi_target_lstm(
        num_countries=num_countries,
        num_features=num_features,
        num_targets=1,
        window_size=window_size
    )
    
    print(model.summary())
    
    import tensorflow as tf
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=[early_stopping],
        verbose=1
    )

    # 9. Evaluation
    print("\n--- Step 9: Evaluation ---")
    results = model.evaluate(test_ds)
    metrics_names = model.metrics_names
    for name, val in zip(metrics_names, results):
        print(f"Test {name}: {val:.6f}")

    print("\nPipeline execution completed!")

if __name__ == "__main__":
    run_full_pipeline()
