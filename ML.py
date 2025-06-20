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
    """Fetch sensor data with PM2.5 support for simulation/fallback purposes."""
    if training_df is not None and not training_df.empty:
        last_row = training_df.iloc[-1]
        return {
            'temperature': last_row['temperature'],
            'humidity': last_row['humidity'],
            'gas': last_row['gas'],
            'pm25': last_row.get('pm25', None),  # Handle missing PM2.5
            'timestamp': datetime.now().isoformat()
        }
    # Fallback to simulated data if no training_df or empty
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
    
    write_header = not os.path.isfile(CSV_FILE_PATH) or os.path.getsize(CSV_FILE_PATH) == 0
    
    with open(CSV_FILE_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow({
            'timestamp': reading['timestamp'],
            'temperature': reading['temperature'],
            'humidity': reading['humidity'],
            'gas': reading['gas'],
            'pm25': reading.get('pm25', '')  # Handle missing PM2.5, store as empty string if None
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
        
        # Drop rows with missing essential values for gas prediction model
        # PM2.5 missing values are acceptable as it's not used for gas prediction
        df.dropna(subset=['temperature', 'humidity', 'gas'], inplace=True) 
        return df
    except Exception as e:
        # print(f"Error loading data: {e}") # Suppress this print during normal operation
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
    """PM2.5 quality assessment based on US EPA standards (simplified)"""
    if pd.isna(pm25_value) or pm25_value < 0: # Handle negative values as well
        return {"label": "Unknown", "class": ""}
    
    if pm25_value <= 12.0:
        return {"label": "Good", "class": "status-good"}
    elif pm25_value <= 35.4:
        return {"label": "Moderate", "class": "status-moderate"}
    elif pm25_value <= 55.4:
        return {"label": "Unhealthy for Sensitive", "class": "status-poor"} # Using status-poor for this range
    elif pm25_value <= 150.4:
        return {"label": "Unhealthy", "class": "status-unhealthy"}
    elif pm25_value <= 250.4:
        return {"label": "Very Unhealthy", "class": "status-danger"} # Use status-danger if you define it
    elif pm25_value <= 350.4:
        return {"label": "Hazardous", "class": "status-hazardous"}
    else:
        return {"label": "Beyond Index", "class": "status-hazardous"} # For extremely high values

def get_overall_air_quality(gas_level, pm25_value):
    """Determine overall air quality based on both gas and PM2.5"""
    # Start with gas quality
    if gas_level < 15:
        overall_label = "Good"
        overall_class = "status-good"
    elif gas_level < 50:
        overall_label = "Moderate"
        overall_class = "status-moderate"
    else:
        overall_label = "Poor"
        overall_class = "status-poor"

    # Now consider PM2.5 quality and potentially worsen the overall status
    if not pd.isna(pm25_value):
        pm25_quality = get_pm25_quality(pm25_value)
        
        # Define a hierarchy for air quality levels (lower index means better quality)
        quality_hierarchy = {
            "Good": 0,
            "Moderate": 1,
            "Unhealthy for Sensitive": 2,
            "Unhealthy": 3,
            "Very Unhealthy": 4,
            "Hazardous": 5,
            "Beyond Index": 6
        }

        current_overall_level = quality_hierarchy.get(overall_label, 0)
        pm25_level = quality_hierarchy.get(pm25_quality['label'], 0)

        if pm25_level > current_overall_level:
            overall_label = pm25_quality['label']
            overall_class = pm25_quality['class']

    return {"label": overall_label, "class": overall_class}


def engineer_features(df):
    """Feature engineering - PM2.5 is NOT used here for gas prediction"""
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['gas_lag1'] = df['gas'].shift(1).fillna(df['gas'].mean())
    df['gas_rolling_mean'] = df['gas'].rolling(window=3, min_periods=1).mean()
    return df

def train_model(df):
    """Model training for gas prediction - PM2.5 is NOT included as a feature"""
    df = engineer_features(df)
    # Ensure only features relevant to gas prediction are used here
    features = ['temperature', 'humidity', 'hour', 'day', 'gas_lag1', 'gas_rolling_mean']
    
    # Drop rows where target variable ('gas') might be NaN after feature engineering (e.g., if there's only one row)
    df.dropna(subset=['gas'], inplace=True) 

    if df.empty or len(df) < 2: # Need at least 2 samples for meaningful split
        print("Warning: Not enough data to train the model effectively.")
        return None, None, features, 0.0 # Return default confidence

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

    print(f"\n=== Model Evaluation (Gas Prediction) ===\nRMSE: {rmse:.2f} ppm\nR²: {r2:.4f}\nConfidence: {confidence:.1f}%")
    return model, scaler, features, confidence

def predict_from_input(file_path, model, scaler, features, training_df):
    """Prediction function with PM2.5 support in output, not prediction"""
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['created_at'])
        df = df.rename(columns={
            'field1': 'temperature', 
            'field2': 'humidity',
            'field4': 'pm25' # Rename field4 to pm25
        })
        
        # Ensure 'pm25' column exists, otherwise fill with NaN
        if 'pm25' not in df.columns:
            df['pm25'] = np.nan
        df['pm25'] = pd.to_numeric(df['pm25'], errors='coerce')


        # Add engineered features required by the model for gas prediction
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day

        # Get last gas readings from training_df for lag features
        last_gas = training_df['gas'].iloc[-1] if not training_df.empty else 0
        df['gas_lag1'] = last_gas
        
        # Ensure gas_rolling_mean is calculated from training_df
        if not training_df.empty and len(training_df) >= 3:
            df['gas_rolling_mean'] = training_df['gas'].rolling(window=3, min_periods=1).mean().iloc[-1]
        else:
            df['gas_rolling_mean'] = last_gas # Fallback if not enough data

        # Get PM2.5 value if available for quality assessment
        pm25_value = df['pm25'].iloc[-1] if 'pm25' in df.columns and not pd.isna(df['pm25'].iloc[-1]) else None

        # Make gas prediction (unchanged ML mechanism, PM2.5 NOT used here)
        # Ensure the input features match what the model was trained on
        input_data_for_scaler = [[df.iloc[-1][f] for f in features]]
        X_input = scaler.transform(input_data_for_scaler)
        
        pred_gas = float(model.predict(X_input)[0])
        pred_gas = np.clip(pred_gas, 0, 1000) # Ensure gas is non-negative and within reasonable bounds
        gas_type = detect_gas_type(pred_gas)
        
        # Determine overall air quality and PM2.5 quality separately
        overall_air_quality = get_overall_air_quality(pred_gas, pm25_value)
        pm25_quality = get_pm25_quality(pm25_value)

        return {
            'gas_level': round(pred_gas, 2),
            'gas_type': gas_type,
            'air_quality': overall_air_quality['label'],
            'pm25_quality': pm25_quality['label'], # Return PM2.5 quality label
            'confidence': 0.85 # Placeholder, use model confidence if available, or pass from train_model
        }
    except Exception as e:
        # print(f"Error during prediction: {e}") # Keep this for debugging
        return {'error': str(e)}

def continuous_monitoring():
    """Monitoring function with PM2.5 support and updated logic"""
    print("Starting air quality monitoring...")
    model, scaler, features, confidence = None, None, None, None
    df = load_data()
    if not df.empty:
        print(f"Loaded {len(df)} existing records.")
        model, scaler, features, confidence = train_model(df)
    else:
        print("No existing data. Starting from scratch.")

    last_retrain_len = len(df) if not df.empty else 0

    try:
        while True:
            # In a real scenario, you'd fetch from hardware. Here, we use a simulated reading.
            # For this integration, the backend is fetching from ThingSpeak, so this section is
            # more for standalone simulation or if ML script was standalone data collector.
            # Since the backend fetches and writes to CSV, this is not strictly needed for the current architecture.
            # We'll rely on the load_data() from the CSV written by the backend.
            
            # This part is for periodic retraining and logging current state
            current_df = load_data()
            if not current_df.empty:
                if len(current_df) > last_retrain_len + RETRAIN_INTERVAL or model is None:
                    print(f"Retraining model with {len(current_df)} records...")
                    model, scaler, features, confidence = train_model(current_df)
                    last_retrain_len = len(current_df)
                
                if model and scaler and not current_df.empty:
                    # This block runs prediction for logging/display purposes in the ML script itself
                    # The actual prediction for the frontend comes from the /api/predict endpoint
                    df_engineered = engineer_features(current_df)
                    if not df_engineered.empty:
                        latest_data_for_prediction = df_engineered.iloc[-1]
                        
                        # Prepare input for scaler, excluding 'pm25' for gas prediction
                        input_features_values = [latest_data_for_prediction[f] for f in features]
                        
                        X_input = scaler.transform([input_features_values])
                        
                        pred_gas = float(model.predict(X_input)[0])
                        pred_gas = np.clip(pred_gas, 0, 1000)
                        
                        gas_type = detect_gas_type(pred_gas)
                        
                        # Get PM2.5 value directly from current_df for its quality assessment
                        latest_pm25 = current_df['pm25'].iloc[-1] if 'pm25' in current_df.columns else np.nan
                        pm25_quality_info = get_pm25_quality(latest_pm25)
                        
                        overall_air_quality_info = get_overall_air_quality(pred_gas, latest_pm25)

                        print(f"\n--- Current Readings & Predictions ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
                        print(f"Temperature: {latest_data_for_prediction['temperature']:.1f}°C")
                        print(f"Humidity: {latest_data_for_prediction['humidity']:.1f}%")
                        print(f"Gas (actual): {latest_data_for_prediction['gas']:.1f} ppm")
                        print(f"PM2.5 (actual): {latest_pm25:.1f} µg/m³ (Quality: {pm25_quality_info['label']})")
                        print(f"Predicted Gas: {pred_gas:.1f} ppm ({gas_type})")
                        print(f"Overall Air Quality: {overall_air_quality_info['label']}")
                        print(f"Model Confidence: {confidence:.1f}%")
                    else:
                        print("No data available after feature engineering for live prediction display.")
                else:
                    print("Model not trained yet or data insufficient for live prediction display.")
            else:
                print("No data loaded from CSV. Waiting for data.")

            time.sleep(SENSOR_READ_INTERVAL) # Wait before next check/retrain trigger
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"An error occurred during continuous monitoring: {e}")

if _name_ == "_main_":
    parser = argparse.ArgumentParser(description="Air Quality Index Monitoring ML Backend")
    parser.add_argument('--predict', type=str, help='Path to CSV file with input data for prediction.')
    parser.add_argument('--data', type=str, default=CSV_FILE_PATH, help='Path to the historical data CSV file.')
    parser.add_argument('--monitor', action='store_true', help='Run in continuous monitoring mode.')

    args = parser.parse_args()

    # Set the CSV_FILE_PATH based on the --data argument for consistency
    CSV_FILE_PATH = args.data

    if args.predict:
        training_df = load_data(args.data) # Load historical data for training context
        if training_df.empty:
            print(json.dumps({'error': 'Historical data not found or empty for training.'}))
        else:
            model, scaler, features, _ = train_model(training_df)
            if model and scaler:
                prediction_result = predict_from_input(args.predict, model, scaler, features, training_df)
                print(json.dumps(prediction_result))
            else:
                print(json.dumps({'error': 'Model could not be trained with historical data.'}))
    elif args.monitor:
        continuous_monitoring()
    else:
        # Default behavior if no specific argument is given, perhaps train and exit or show help
        parser.print_help()
        print("\nTo run the server, ensure the Node.js backend calls this script with --predict or --monitor.")
