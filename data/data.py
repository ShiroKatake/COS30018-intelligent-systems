import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import csv

timeInterval = 15

def get_all_scats_neighbors(data):
    df = pd.read_csv(data, encoding='utf-8').fillna(0)
    scats_neighbours_dict = {}

    for _, row in df.iterrows():
        scats_number = row['SCATS Number']
        neighbors = row['NEIGHBOURS']
        lat = row['NB_LATITUDE']
        long = row['NB_LONGITUDE']
        neighbors_list = neighbors.split(';') if neighbors else []
        scats_neighbours_dict[scats_number] = {
            'lat': lat,
            'long': long,
            'neighbors': neighbors_list
        }

    return scats_neighbours_dict

def get_all_scats_points(filename):
    unique_values = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)

        # Skip the first row (header)
        next(csvreader)
        
        # Iterate through each row in the CSV
        for row in csvreader:
            if row:  # Check if the row is not empty
                # Extract the value from the first column
                value = row[0]
                
                # Check if the value is not already in the list
                if value not in unique_values:
                    unique_values.append(value)
    print(unique_values)
    return unique_values

def get_lat_long_from_scats(data, scats):
    # Read data
    df = pd.read_csv(data, encoding='utf-8').fillna(0)

    # Filter the DataFrame based on the SCATS number. Makes a new dataframe thats just got the values for that SCATS number
    filtered_df = df[df['SCATS Number'] == int(scats)]

    # Get the lat and long from the new table
    lat = filtered_df['NB_LATITUDE'].iloc[0]
    long = filtered_df['NB_LONGITUDE'].iloc[0]

    return lat, long

def process_data(train, test, lags):

    # Reading the CSV files and filling NaN with 0
    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    #df2 = pd.read_csv(test, encoding='utf-8').fillna(0)

    #Get the position of the first time column in the CSV
    columnPositionOfTimeColumn = df1.columns.get_loc("V00")

    flow1 = df1.to_numpy()[:, columnPositionOfTimeColumn:]
    
    numberOfTimeColumns = flow1.shape[1]

    minutesInData = timeInterval * numberOfTimeColumns


    nb_latitude = df1['NB_LATITUDE'].to_numpy().reshape(-1, 1)
    nb_longitude = df1['NB_LONGITUDE'].to_numpy().reshape(-1, 1)

    # Scaling using MinMaxScaler
    flowScaler = MinMaxScaler(feature_range=(0, 1)).fit(flow1)
    latitudeScaler = MinMaxScaler(feature_range=(0, 1)).fit(nb_latitude)
    longitudeScaler = MinMaxScaler(feature_range=(0, 1)).fit(nb_longitude)

    # Transform
    flowScaled = flowScaler.transform(flow1)
    nb_latitude = latitudeScaler.transform(nb_latitude)
    nb_longitude = longitudeScaler.transform(nb_longitude)
    
    # Concatenate both lat and long into one
    latitudeAndLongitude = np.concatenate((nb_latitude, nb_longitude), axis=1)

    # Initializing empty lists for train and test data
    train, test = [], []

    # Loop over the rows in the flowScaled 2D array.
    for rowIndex in range(len(flowScaled)):
        
        # Loop over the number of time columns specified.
        for timeColumn in range(numberOfTimeColumns):
            
            # If we're at the last time column, set the index to -1 (likely to get the last column).
            # Otherwise, use the current timeColumn index.
            flowColumnIndex = timeColumn if timeColumn != numberOfTimeColumns - 1 else -1
            
            # Scale the timeColumn value based on timeInterval and minutesInData.
            timeScaled = timeColumn * timeInterval / minutesInData
            
            # Fetch the latitude and longitude for the current row.
            latLong = latitudeAndLongitude[rowIndex]
            
            # Fetch the flow value for the current row and specified column.
            # We add 1 to the flowColumnIndex, as the actual data starts one column ahead.
            flowValue = flowScaled[rowIndex, flowColumnIndex + 1]
            
            # Concatenate the timeScaled, latLong, and flowValue to form a single row of inputData.
            inputData = np.concatenate([[timeScaled], latLong, [flowValue]])

            # Append this new row of inputData to the train list.
            train.append(inputData)

    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    X_train = train[:, :-1]
    y_train = train[:, -1]

    X_test = None
    y_test = None

    # X_test = test[:, :-1]
    # y_test = test[:, -1]


    return X_train, y_train, X_test, y_test, flowScaler, latitudeScaler, longitudeScaler
