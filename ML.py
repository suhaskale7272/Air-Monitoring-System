import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import json
import argparse
import os
import csv
import time
from datetime import datetime

# Constants
SENSOR_READ_INTERVAL = 60
CSV_FILE_PATH = "data/sensor_readings.csv"
RETRAIN_INTERVAL = 10

def fetch_sensor_data(training_df=None):
    """Fetch sensor data with PM2.5 support"""
    if training_df is not None and not training_df.empty:
        last_row = training_df.iloc[-1]
        return {
            'temperature': last_row['temperature'],
            'humidity': last_row['humidity'],
            'gas': last_row['gas'],
            'pm25': last_row.get('pm25', None),  # Handle missing PM2.5
            'timestamp': datetime.now().isoformat()
        }
    # Fallback to simulated data
    temperature = round(np.random.uniform(20, 35), 2)
    humidity = round(np.random.uniform(30, 60), 2)
    gas = round(np.random.uniform(4, 15), 2)
    pm25 = round(np.random.uniform(5, 50), 2)  # Simulated PM2.5
    return {
        'temperature': temperature,
        'humidity': humidity,
        'gas': gas,
        'pm25': pm25,
        'timestamp': datetime.now().isoformat()
    }

def store_sensor_reading(reading):
    """Store sensor readings with PM2.5 support"""
    os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
    header = ['timestamp', 'temperature', 'humidity', 'gas', 'pm25']
    
    write_header = not os.path.isfile(CSV_FILE_PATH)
    
    with open(CSV_FILE_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({
            'timestamp': reading['timestamp'],
            'temperature': reading['temperature'],
            'humidity': reading['humidity'],
            'gas': reading['gas'],
            'pm25': reading.get('pm25', '')  # Handle missing PM2.5
        })

def load_data(file_path=CSV_FILE_PATH):
    """Load data with PM2.5 support and better error handling"""
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Convert numeric columns and handle missing values
        numeric_cols = ['temperature', 'humidity', 'gas', 'pm25']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing essential values
        df.dropna(subset=['temperature', 'humidity', 'gas'], inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def detect_gas_type(concentration):
    """Gas type detection - unchanged as requested"""
    if concentration < 12:
        return "Clean Air"
    elif 12 <= concentration < 20:
        return "CO2 (Carbon Dioxide)"
    elif 20 <= concentration < 30:
        return "NO2 (Nitrogen Dioxide)"
    elif 30 <= concentration < 45:
        return "NH3 (Ammonia)"
    elif 45 <= concentration < 60:
        return "CO (Carbon Monoxide)"
    else:
        return "Mixed Pollutants"

def get_pm25_quality(pm25_value):
    """PM2.5 quality assessment"""
    if pd.isna(pm25_value):
        return None
    
    if pm25_value <= 12:
        return {"label": "Good", "class": "status-good"}
    elif pm25_value <= 35:
        return {"label": "Moderate", "class": "status-moderate"}
    elif pm25_value <= 55:
        return {"label": "Unhealthy for Sensitive", "class": "status-poor"}
    elif pm25_value <= 150:
        return {"label": "Unhealthy", "class": "status-unhealthy"}
    else:
        return {"label": "Hazardous", "class": "status-hazardous"}

def engineer_features(df):
    """Feature engineering - unchanged as requested"""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['gas_lag1'] = df['gas'].shift(1).fillna(df['gas'].mean())
    df['gas_rolling_mean'] = df['gas'].rolling(window=3, min_periods=1).mean()
    return df

def train_model(df):
    """Model training - unchanged as requested"""
    df = engineer_features(df)
    features = ['temperature', 'humidity', 'hour', 'day', 'gas_lag1', 'gas_rolling_mean']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df['gas'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, verbosity=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    confidence = max(0, min(100, 100 * r2))

    print(f"\n=== Model Evaluation ===\nRMSE: {rmse:.2f} ppm\nR²: {r2:.4f}\nConfidence: {confidence:.1f}%")
    return model, scaler, features, confidence

def predict_from_input(file_path, model, scaler, features, training_df):
    """Prediction function with PM2.5 support in output"""
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['created_at'])
        df = df.rename(columns={
            'field1': 'temperature', 
            'field2': 'humidity',
            'field4': 'pm25'
        })
        
        # Add engineered features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        last_gas = training_df['gas'].iloc[-1] if not training_df.empty else 0
        df['gas_lag1'] = last_gas
        df['gas_rolling_mean'] = training_df['gas'].rolling(window=3, min_periods=1).mean().iloc[-1] if not training_df.empty else last_gas

        # Get PM2.5 value if available
        pm25_value = df['pm25'].iloc[-1] if 'pm25' in df.columns and not pd.isna(df['pm25'].iloc[-1]) else None

        # Make prediction (unchanged ML mechanism)
        X_input = scaler.transform([[df.iloc[-1][f] for f in features]])
        pred_gas = float(model.predict(X_input)[0])
        pred_gas = np.clip(pred_gas, 0, 1000)
        gas_type = detect_gas_type(pred_gas)
        
        # Determine air quality based on both gas and PM2.5
        air_quality = "Good"
        if pred_gas >= 20 or (pm25_value and pm25_value > 35):
            air_quality = "Poor"
        elif pm25_value and pm25_value > 55:
            air_quality = "Unhealthy"
            
        return {
            'gas_level': pred_gas,
            'gas_type': gas_type,
            'air_quality': air_quality,
            'pm25_quality': get_pm25_quality(pm25_value),
            'confidence': 0.85  # Placeholder, use model confidence if available
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {'error': str(e)}

def continuous_monitoring():
    """Monitoring function with PM2.5 support"""
    print("Starting air quality monitoring...")
    model, scaler, features, confidence = None, None, None, None
    df = load_data()
    if not df.empty:
        print(f"Loaded {len(df)} existing records.")
        model, scaler, features, confidence = train_model(df)
    else:
        print("No existing data. Starting from scratch.")

    try:
        while True:
            reading = fetch_sensor_data(df)
            store_sensor_reading(reading)
            df = load_data()
            if len(df) % RETRAIN_INTERVAL == 0 or model is None:
                model, scaler, features, confidence = train_model(df)

            if model and scaler:
                df = engineer_features(df)
                latest = df.iloc[-1]
                X_input = scaler.transform([[latest[f] for f in features]])
                pred_gas = float(model.predict(X_input)[0])
                pred_gas = np.clip(pred_gas, 0, 1000)
                gas_type = detect_gas_type(pred_gas)
                pm25_quality = get_pm25_quality
