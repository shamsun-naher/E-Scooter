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

# Load data
X_forecasting = pd.read_csv('data.csv')
train_pred_df = pd.read_csv('train_pred_df.csv')
# Define training set
train_df = X_forecasting[X_forecasting['y'].notna()].copy()

# Fit Prophet model (cached)
#prophet_hourly = fit_prophet_model(train_df)

# Predict over all dates
#train_pred_df = prophet_hourly.predict(X_forecasting)

# Add baseline predictions from Prophet (using rolling mean)
X_forecasting['baseline'] = train_pred_df['yhat'].rolling(window=74).mean()

# Drop rows where 'baseline' is NaN
X_forecasting.dropna(subset=['baseline'], inplace=True)
X_forecasting.set_index('ds', inplace=True)

# Prepare for XGBoost
X_train = X_forecasting[X_forecasting['y'].notna()].copy()
y_train = X_train.pop('y')

# Define XGBoost model
#boost_model = xgb.XGBRegressor(
   # n_estimators=10000,  # Reduce number of estimators
   # learning_rate=0.01,
   # max_depth=5,
   # subsample=0.8,
   # colsample_bytree=0.8,
   # reg_alpha=0.1,
   # reg_lambda=1.0
#)

# Fit XGBoost model
#boost_model.fit(X_train, y_train)
# Save the trained model
#joblib.dump(boost_model, "boost_model.pkl")

# Load the trained model
boost_model = joblib.load("boost_model.pkl")
# Predict on the full forecasting DataFrame
predictions = boost_model.predict(X_forecasting.drop(columns=['y']))

# Feature importances
feature_importance = boost_model.feature_importances_
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Streamlit UI
st.write(importance_df)
# Future Forecast Plot
from plotly.subplots import make_subplots
fig= make_subplots(rows=1, cols=1,
                           subplot_titles=["Important Features Selection"])
# Plot feature importance
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(importance_df['feature'], importance_df['importance'])
ax.invert_yaxis()  # largest importance on top
ax.set_xlabel("Feature Importance")
ax.set_title("XGBoost Feature Importances")
#fig.show()
# Save plot to buffer
#buf = io.BytesIO()
#fig.savefig(buf, format="png")
#buf.seek(0)

# Display plot in Streamlit
st.pyplot(fig)

# Provide a download link for the figure
#st.download_button(
    #label="Download Figure",
    #data=buf,
    #file_name="feature_importance.png",
    #mime="image/png"
#)