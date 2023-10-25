import math
import sys
import json
import argparse
from datetime import datetime
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from data.data import get_scats_dict
from data.data import get_lat_long_from_scats, get_all_scats_points
from keras.models import load_model
from keras.utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt

from map.map import get_routes

warnings.filterwarnings("ignore")

file1 = 'data/myData.csv'
file2 = 'data/myData2.csv'
timeInterval = 15


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape

def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """

    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)


def plot_results(y_true, y_preds, names):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    plt.show()

def round_to_nearest_15(minutes):
    """Rounds the time to the nearest 15 minutes."""
    remainder = minutes % 15
    if remainder < 7.5:
        return minutes - remainder
    else:
        return minutes + (15 - remainder)

def minute_to_time(minute_of_day):
    hours = minute_of_day // 60
    minutes = minute_of_day % 60
    return "{:02d}:{:02d}".format(hours, minutes)

def get_previous_11_data(df, lat, lon, date, time):
    # Convert
    time = minute_to_time(time)
    
    # Date to Day of week
    day = date.strftime('%A')

    # Fetch the previous 11 data points
    data_points = []
    for _ in range(11):
        # Decrement time
        col_index = list(df.columns).index(time) - 1
        
        # If we've reached the beginning of the day
        if col_index < 3:
            # Move to the previous day
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day = days[days.index(day) - 1]
            time = "23:45"
            col_index = -1  # Reset to the last column (23:45)
        else:
            time = df.columns[col_index]

        # Find the row corresponding to the given latitude, longitude, and day
        row = df[(df['NB_LATITUDE'] == lat) & (df['NB_LONGITUDE'] == lon) & (df['Date'] == day)]
        
        # If no such row exists, raise an error
        if row.empty:
            raise ValueError(f"No data found for the given location and day {day}.")
        
        data_points.insert(0, row.iloc[0, col_index])  # prepend the data point

    # Add a dummy value to be removed later. This is needed because the scaler is expecting 12 features, but we need to use 11 to predict, so we'll append one and then scale and then remove it.
    data_points = data_points + [0]

    # scale with dummy
    scaled_data_points = flow_scaler.transform(np.array(data_points).reshape(1, -1))

    # remove dummy value
    scaled_data_points = scaled_data_points[:, :-1]


    #print(f"SCALED: {scaled_data_points}")

    return np.array(scaled_data_points)

def predict_traffic_flow(latitude, longitude, time, date, model):
    # Parse the date
    date = datetime.strptime(date, '%d/%m/%Y')
    day_of_week = date.weekday()

    # Custom order based on your sequence
    custom_order = [4, 0, 5, 6, 3, 1, 2]

    # Create the binary list based on the custom order
    binary_list = [1 if custom_order[i] == day_of_week else 0 for i in range(7)]

    time = round_to_nearest_15(time)

    # Normalize the time by dividing it by the total minutes in a day (1440)
    normalized_time = time / 1440 # This number should be same as df['Time'] in data.py

    # Transform latitude and longitude using respective scalers
    scaled_latitude = lat_scaler.transform(np.array(latitude).reshape(1, -1))[0][0]
    scaled_longitude = long_scaler.transform(np.array(longitude).reshape(1, -1))[0][0]

    

    # Prepare test data
    #x_test = np.array([[scaled_latitude, scaled_longitude, 0, 0, 0, 0, 0, 0, 1, 0,0,0,0,0,0,0,0,0,0,0]])
    #print(f"TARGET: {x_test}")

    x_test = np.array([[scaled_latitude, scaled_longitude]])
    for day in binary_list:
        x_test = np.append(x_test, [day]).reshape(1, -1)
    
    x_test = np.append(x_test, get_previous_11_data(process_data_csv, latitude, longitude, date, time)).reshape(1, -1)

    #print(f"x_test: {x_test}")

    #print(f"FIRST x_test SHAPE: {x_test.shape}")
    #print(x_test.shape)
    # Reshape x_test based on the chosen model
    if model in ['SAEs']:
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
    else:
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    #print(f"SECOND x_test SHAPE: {x_test.shape}")

    # Map the string name of the model to the actual model object
    model_map = {
        'lstm': lstm,
        'gru': gru,
        'saes': saes,
        'nn': nn
    }

    # Select the desired model
    selected_model = model_map.get(model.lower())
    if selected_model is None:
        raise ValueError(f"Unsupported model: {model}")

    # Predict using the selected model
    predicted = selected_model.predict(x_test)

    #print(f"SHAPE = {predicted.shape}")

    predicted_structure = np.zeros(shape=(len(predicted), 12))

    predicted_structure[:, 0] = predicted.reshape(-1, 1)[:, 0]

    final_prediction = flow_scaler.inverse_transform(predicted_structure)[:, 0].reshape(1, -1)[0][0]
    

    return final_prediction


# Just temporarily this isnt doing anything
def main():
    lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    saes = load_model('model/saes.h5')
    nn = load_model('model/nn.h5')
    models = [lstm, gru, saes, nn]
    names = ['LSTM', 'GRU', 'SAEs', 'My model']

    lag = 12
    file1 = 'data/train.csv'
    file2 = 'data/test.csv'
    _, _, X_test, y_test, scaler = process_data(file1, file2, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

    y_preds = []
    for name, model in zip(names, models):
        if name == 'SAEs' or name == 'My model':
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        file = 'images/' + name + '.png'
        plot_model(model, to_file=file, show_shapes=True)
        predicted = model.predict(X_test)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
        y_preds.append(predicted[:288])
        print(name)
        eva_regress(y_test, predicted)

    plot_results(y_test[: 288], y_preds, names)


def initialise_models():
    global lstm
    global gru
    global saes
    global nn
    global flow_scaler
    global lat_scaler
    global long_scaler
    global X_train_global
    global y_train_global
    global process_data_csv

    lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    saes = load_model('model/saes.h5')
    nn = load_model('model/nn.h5')
    X_train_global, y_train_global, flow_scaler, lat_scaler, long_scaler = process_data()
    process_data_csv = pd.read_csv('data/ProcessedData.csv', encoding='utf-8').fillna(0)
    print("INITIALISED")

def time_string_to_minute_of_day(time_str):
    # Split the time string by the colon to get the hour and minute parts.
    hour_str, minute_str = time_str.split(":")
    
    # Convert the hour and minute parts to integers.
    hour = int(hour_str)
    minute = int(minute_str)
    
    # Calculate the minute of the day.
    minute_of_day = hour * 60 + minute
    
    return minute_of_day

if __name__ == '__main__':
    initialise_models()
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start_scat",
        default="970",
        help="Start scat number (default 970).")
    parser.add_argument(
        "--end_scat",
        default="2820",
        help="End scat number (default 2820).")
    parser.add_argument(
        "--date",
        default="14/10/2023",
        help="Date to predict (dd/mm/yyyy).")
    parser.add_argument(
        "--time",
        default="0:30",
        help="Time to predict (hh:mm).")
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to use for prediction (lstm, gru, saes, nn [default]).")
    args = parser.parse_args()

    scat_data = get_scats_dict("data/SCATS_SITE_LISTING.csv")
    result = {}

    for scat in scat_data:
        lat, long = get_lat_long_from_scats(file1, scat)

        # Make prediction
        flow_prediction = predict_traffic_flow(latitude=lat, longitude=long, date=args.date, time=time_string_to_minute_of_day(args.time), model=args.model)
        print(f'{scat}: {flow_prediction}') # TO BE COMMENTED OUT WHEN NOT TESTING
        scat_data[scat].flow = flow_prediction


    # routes = get_routes(scat_data, args.start_scat, args.end_scat)
    # response = routes
    
    # print(json.dumps(response))
    # sys.stdout.flush()
