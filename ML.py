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
import time  # Added the missing import for time
from datetime import datetime

# Constants
SENSOR_READ_INTERVAL = 60
CSV_FILE_PATH = "data/sensor_readings.csv"
RETRAIN_INTERVAL = 10

def fetch_sensor_data(training_df=None):
    if training_df is not None and not training_df.empty:
        last_row = training_df.iloc[-1]
        return {
            'temperature': last_row['temperature'],
            'humidity': last_row['humidity'],
            'gas': last_row['gas'],
            'timestamp': datetime.now().isoformat()
        }
    # Fallback to simulated data
    temperature = round(np.random.uniform(20, 35), 2)
    humidity = round(np.random.uniform(30, 60), 2)
    gas = round(np.random.uniform(4, 15), 2)
    return {
        'temperature': temperature,
        'humidity': humidity,
        'gas': gas,
        'timestamp': datetime.now().isoformat()
    }

def store_sensor_reading(reading):
    os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
    if not os.path.isfile(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'temperature', 'humidity', 'gas'])

    with open(CSV_FILE_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            reading['timestamp'],
            reading['temperature'],
            reading['humidity'],
            reading['gas']
        ])

def load_data(file_path=CSV_FILE_PATH):
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['temperature'] = pd.to_numeric(df['temperature'], errors='coerce')
        df['humidity'] = pd.to_numeric(df['humidity'], errors='coerce')
        df['gas'] = pd.to_numeric(df['gas'], errors='coerce')
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def detect_gas_type(concentration):
    if concentration < 8:
        return "Clean Air"
    elif 8 <= concentration < 20:
        return "CO2 (Carbon Dioxide)"
    elif 20 <= concentration < 30:
        return "NO2 (Nitrogen Dioxide)"
    elif 30 <= concentration < 45:
        return "NH3 (Ammonia)"
    elif 45 <= concentration < 60:
        return "CO (Carbon Monoxide)"
    else:
        return "Mixed Pollutants"

def engineer_features(df):
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['gas_lag1'] = df['gas'].shift(1).fillna(df['gas'].mean())
    df['gas_rolling_mean'] = df['gas'].rolling(window=3, min_periods=1).mean()
    return df

def train_model(df):
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
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['created_at'])
        df = df.rename(columns={'field1': 'temperature', 'field2': 'humidity'})
        
        # Add engineered features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        last_gas = training_df['gas'].iloc[-1] if not training_df.empty else 0
        df['gas_lag1'] = last_gas
        df['gas_rolling_mean'] = training_df['gas'].rolling(window=3, min_periods=1).mean().iloc[-1] if not training_df.empty else last_gas

        X_input = scaler.transform([[df.iloc[-1][f] for f in features]])
        pred_gas = float(model.predict(X_input)[0])
        pred_gas = np.clip(pred_gas, 0, 1000)
        gas_type = detect_gas_type(pred_gas)
        return {
            'gas_level': pred_gas,
            'gas_type': gas_type,
            'air_quality': 'Good' if pred_gas < 20 else 'Poor',
            'confidence': 0.85  # Placeholder, use model confidence if available
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {'error': 'Prediction failed'}

def continuous_monitoring():
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

                print(f"\n[{datetime.now().isoformat()}] Latest:")
                print(f"Temp: {latest['temperature']} °C, Humidity: {latest['humidity']} %")
                print(f"Predicted Gas: {pred_gas:.2f} ppm - {gas_type}")
                print(f"Confidence Score: {confidence:.1f}%")

            time.sleep(SENSOR_READ_INTERVAL)
    except KeyboardInterrupt:
        print("Monitoring stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', type=str, help='Path to prediction input CSV')
    parser.add_argument('--data', type=str, help='Path to training data CSV')
    parser.add_argument('--retrain', type=str, help='Path to training data CSV for retraining')
    args = parser.parse_args()

    if args.predict and args.data:
        df = load_data(args.data)
        if not df.empty:
            model, scaler, features, confidence = train_model(df)
            result = predict_from_input(args.predict, model, scaler, features, df)
            print(json.dumps(result))
        else:
            print(json.dumps({'error': 'No training data available'}))
    elif args.retrain:
        df = load_data(args.retrain)
        if not df.empty:
            model, scaler, features, confidence = train_model(df)
            print(json.dumps({'status': 'Model retrained', 'confidence': confidence}))
        else:
            print(json.dumps({'error': 'No training data available'}))
    else:
        continuous_monitoring()
