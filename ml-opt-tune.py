import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Load dataset (assuming it's available)
lime_data = pd.read_csv('lime_data.csv', nrows=100000)

# Convert 'Start Time' to datetime
lime_data['Start Time'] = pd.to_datetime(lime_data['Start Time'], errors='coerce')

# Extract date-related features
lime_data["month"] = lime_data["Start Time"].dt.month
lime_data["weekday"] = lime_data["Start Time"].dt.weekday
lime_data["hour"] = lime_data["Start Time"].dt.hour

# Group by time features
spt_daily_trips = lime_data.groupby(['hour', 'weekday', 'month']).agg(trip_count=('Trip ID', 'count')).reset_index()

# Streamlit layout
st.set_page_config(page_title='Trip Forecasting App', layout='wide')
st.write("## Trip Forecasting with Machine Learning")

# Sidebar parameters
split_size = st.sidebar.slider('Train/Test Split (%)', 10, 90, 80, 5)
n_estimators_range = st.sidebar.slider('Number of Estimators', 10, 200, (10, 50), 10)
max_features_range = st.sidebar.slider('Max Features', 1, 5, (1, 3), 1)
min_samples_split = st.sidebar.slider('Min Samples Split', 2, 10, 2, 1)
min_samples_leaf = st.sidebar.slider('Min Samples Leaf', 1, 10, 2, 1)

# Convert tuple to range
n_estimators_values = list(range(n_estimators_range[0], n_estimators_range[1] + 1, 10))
max_features_values = list(range(max_features_range[0], max_features_range[1] + 1))

param_grid = {
    'n_estimators': n_estimators_values,
    'max_features': max_features_values
}

# Define function to build and train model
def build_model(df):
    X = df[['hour', 'weekday', 'month']]
    y = df['trip_count']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_size / 100), random_state=42)

    rf = RandomForestRegressor(
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        bootstrap=True
    )

    grid = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)

    y_pred = grid.best_estimator_.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    st.write(f"### Model Performance")
    st.write(f"Best Parameters: {grid.best_params_}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")

    # Generate Hyperparameter Tuning Plot
    grid_results = pd.DataFrame(grid.cv_results_)
    params_df = pd.json_normalize(grid_results['params'])
    grid_results = pd.concat([grid_results, params_df], axis=1)

    grid_pivot = grid_results.pivot_table(index='max_features', columns='n_estimators', values='mean_test_score')

    x = grid_pivot.columns.values
    y = grid_pivot.index.values
    z = grid_pivot.values
     #-----Plot-----#
    layout = go.Layout(
            xaxis=go.layout.XAxis(
              title=go.layout.xaxis.Title(
              text='n_estimators')
             ),
             yaxis=go.layout.YAxis(
              title=go.layout.yaxis.Title(
              text='max_features')
            ) )
    fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
    fig.update_layout(title='Hyperparameter tuning',
                      scene = dict(
                        xaxis_title='n_estimators',
                        yaxis_title='max_features',
                        zaxis_title='R2'),
                      autosize=False,
                      width=800, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)


    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title='Hyperparameter Tuning', width=800, height=800)
    st.plotly_chart(fig)

# Run the model
build_model(spt_daily_trips)