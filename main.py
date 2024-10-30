# main.py



#=====================================================
#  Suppress Other Warnings and Clean Output
#=====================================================

import os
import warnings
import absl.logging
import tensorflow as tf

# Set environment variables to disable CUDA, minimize TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Suppress Abseil logging for CUDA plugin registrations
absl.logging.set_verbosity('error')

# Set TensorFlow logging to show only critical issues
tf.get_logger().setLevel('ERROR')

# Suppress general UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# Optionally suppress all warnings (if needed)
warnings.filterwarnings("ignore")

#=====================================================



from src.data_loader import get_combined_data
from src.preprocessing import preprocess_data
from src.models import run_ets, run_arima, run_linear_regression, run_lstm



def main():
    print("=========== Menu ===========")
    print("1- Fetch Data")
    print("2- LSTM model")
    print("3- ARIMA model")
    print("4- Linear Regression")
    print("5- ETS")
    print("6- Quit")
    print("============================")

    combined_data = None
    scaled_data = None
    scaler = None  # Define scaler variable to store scaler object

    while True:
        choice = input("Select an option: ")
        
        if choice == '1':
            combined_data = get_combined_data()
            if not combined_data.empty:
                print("[INFO] Data fetched successfully.")
                scaled_data, scaler = preprocess_data(combined_data)
            else:
                print("[ERROR] Failed to fetch data.")
                
        elif choice == '2' and scaled_data is not None:
            forecast, mse = run_lstm(scaled_data, scaler)  # Pass scaler as an argument
            print(f"[LSTM] Forecast: {forecast}, MSE: {mse}")
            
        elif choice == '3' and combined_data is not None:
            forecast, mse = run_arima(combined_data['Close'])
            print(f"[ARIMA] Forecast: {forecast}, MSE: {mse}")
            
        elif choice == '4' and combined_data is not None:
            forecast, mse = run_linear_regression(combined_data)
            print(f"[Linear Regression] Forecast: {forecast}, MSE: {mse}")
            
        elif choice == '5' and combined_data is not None:
            forecast, mse = run_ets(combined_data['Close'])
            print(f"[ETS] Forecast: {forecast}, MSE: {mse}")
            
        elif choice == '6':
            print("Exiting program.")
            break
            
        else:
            print("Invalid choice or data not loaded. Please fetch data first or choose a valid option.")

if __name__ == "__main__":
    main()
