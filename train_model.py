import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import streamlit as st
from prophet import Prophet
from sklearn.model_selection import train_test_split
import io
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

# Cache to avoid repeated fitting of Prophet model
@st.cache_resource
def fit_prophet_model(train_df):
    prophet_hourly = Prophet(
        growth='logistic', 
        n_changepoints=30,
        changepoint_range=0.9,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='multiplicative',
        seasonality_prior_scale=5.0,
        changepoint_prior_scale=0.05,
        interval_width=0.95,
        uncertainty_samples=1000
    )
    prophet_hourly.fit(train_df)
    return prophet_hourly

# Load data
X_forecasting = pd.read_csv('data.csv')

# Define training set
train_df = X_forecasting[X_forecasting['y'].notna()].copy()

# Fit Prophet model (cached)
prophet_hourly = fit_prophet_model(train_df)

# Predict over all dates
train_pred_df = prophet_hourly.predict(X_forecasting)

# Add baseline predictions from Prophet (using rolling mean)
X_forecasting['baseline'] = train_pred_df['yhat'].rolling(window=74).mean()

# Drop rows where 'baseline' is NaN
X_forecasting.dropna(subset=['baseline'], inplace=True)
X_forecasting.set_index('ds', inplace=True)

# Prepare for XGBoost
X_train = X_forecasting[X_forecasting['y'].notna()].copy()
y_train = X_train.pop('y')

# Define XGBoost model
boost_model = xgb.XGBRegressor(
    n_estimators=5000,  # Reduce number of estimators
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0
)

# Fit XGBoost model
boost_model.fit(X_train, y_train)
# Save the trained model
joblib.dump(boost_model, "boost_model.pkl")
print("Model saved successfully!")