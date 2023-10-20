import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
from data.data import get_scats_dict
from keras.models import load_model
from keras.utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt

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

def predict_traffic_flow(latitude, longitude, time: float, model: str):
    # Normalize the time by dividing it by the total minutes in a day (1440)
    normalized_time = time / 1440 #this number should be same as minutesInData variable from data.py
    
    # Transform latitude and longitude using respective scalers
    scaled_latitude = lat_scaler.transform(np.array(latitude).reshape(1, -1))[0][0]
    scaled_longitude = long_scaler.transform(np.array(longitude).reshape(1, -1))[0][0]
    
    # Prepare test data
    X_test = np.array([[normalized_time, scaled_latitude, scaled_longitude]])
    
    # Reshape X_test based on the chosen model
    if model in ['SAEs', 'nn']:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    else:
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

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

    print(f"Select {model}")

    # Predict using the selected model
    predicted = selected_model.predict(X_test)

    # Create a new array to structure the predictions
    predicted_structure = np.zeros(shape=(len(predicted), 96))
    predicted_structure[:, 0] = predicted.reshape(-1, 1)[:, 0]

    # Transform the prediction using the y_scaler to get the actual prediction
    final_prediction = y_scaler.inverse_transform(predicted_structure)[:, 0].reshape(1, -1)[0][0]
    
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
    global y_scaler
    global lat_scaler
    global long_scaler

    lstm = load_model('model/lstm.h5')
    gru = load_model('model/gru.h5')
    saes = load_model('model/saes.h5')
    nn = load_model('model/nn.h5')
    _, _, _, _, y_scaler, lat_scaler, long_scaler = process_data(file1, '', 0)

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
    
    scat_data = get_scats_dict("data/SCATS_SITE_LISTING.csv")
    
    start_scat = input("What scat number to start from? (e.g. 970): ") or "970"
    end_scat = input("What scat number to end at? (e.g. 2827): ") or "2827"
    # date = input("What date to predict? (e.g. 2016-3-4): ") # Convert this to just the day of the month to output true data of oct to compare
    time = input("What time to predict? (e.g. 14:30): ") or "14:30"
    model = input("What model to use? (lstm, gru, saes, nn): ") or "nn"

    for scat in scats_points:
        lat, long = get_lat_long_from_scats(file1, scat)

        # Make prediction
        lstmPrediction = str(predict_traffic_flow(latitude=lat, longitude=long, time=time_string_to_minute_of_day(time), model='lstm'))
        gruPrediction = str(predict_traffic_flow(latitude=lat, longitude=long, time=time_string_to_minute_of_day(time), model='gru'))
        saesPrediction = str(predict_traffic_flow(latitude=lat, longitude=long, time=time_string_to_minute_of_day(time), model='saes'))
        nnPrediction = str(predict_traffic_flow(latitude=lat, longitude=long, time=time_string_to_minute_of_day(time), model='nn'))

        result[scat] = {
            'lstm': lstmPrediction,
            'gru': gruPrediction,
            'saes': saesPrediction,
            'nn': nnPrediction
        }

    print(result)
