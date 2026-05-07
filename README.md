# Economic Forecasting Project

A machine learning project for forecasting key macroeconomic indicators using historical economic data and advanced time-series modeling techniques.

This project focuses on predicting indicators such as GDP, Inflation Rate, and Unemployment Rate through feature engineering, preprocessing pipelines, and deep learning / machine learning models.

---

# Project Overview

The goal of this project is to build an intelligent forecasting system capable of analyzing macroeconomic trends and predicting future economic conditions.

The project includes:

- Data preprocessing pipeline
- Feature engineering
- Time-series preparation
- Exploratory Data Analysis (EDA)
- Machine Learning & Deep Learning models
- Multi-output forecasting
- Model evaluation & visualization

---

# Features

- Clean and modular pipeline structure
- Handling missing values and outliers
- Lag features & moving averages
- Scaling using `RobustScaler`
- Correlation analysis
- Feature importance analysis using XGBoost
- Multi-output forecasting
- Sequence preparation for LSTM models
- Visualization of trends and predictions

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- Matplotlib
- Seaborn
- Streamlit

---

# Project Structure

```bash
macro_data_project/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Modeling.ipynb
│
├── src/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── modeling/
│   ├── visualization/
│
├── app/
│   ├── streamlit_app.py
│
├── models/
│
├── requirements.txt
│
└── README.md
```

---

# Target Variables

The project forecasts:

- GDP
- Inflation Rate
- Unemployment Rate

---

# Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Removing duplicates
- Outlier treatment
- Feature scaling using `RobustScaler`
- Train / Validation / Test split
- Log transformations (`log1p`)
- Sequence window generation for LSTM

---

# Feature Engineering

Several engineered features were created to improve forecasting performance:

- Lag Features (`GDP_lag1`)
- Moving Averages (`GDP_MA3`)
- Difference Features
- Economic Ratios
- Exchange Volatility
- Trade-to-GDP Ratio

Feature importance analysis was performed using XGBoost.

---

# Models Used

## Machine Learning

- XGBoost Regressor
- Random Forest
- Linear Regression

## Deep Learning

- LSTM (Long Short-Term Memory)

---

# Model Evaluation

Evaluation metrics used:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Visualization includes:

- Actual vs Predicted graphs
- Correlation heatmaps
- Feature importance charts

---

# How to Run

## 1. Clone the repository

```bash
git clone <your-repository-link>
cd macro_data_project
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

# Future Improvements

- Add Transformer-based forecasting models
- Deploy the project on cloud platforms
- Real-time economic data integration
- Hyperparameter optimization
- Interactive dashboards

---

# Author

Ahmed / Omar Reyad Shohieb  
Machine Learning Engineer | AI Enthusiast

---

# Notes

This project was built for educational and research purposes to explore how AI and machine learning can be applied to macroeconomic forecasting and financial analysis.
