import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import csv

timeInterval = 15

class Scat:
    def __init__(self, number, name, lat, long, flow, neighbors):
        self.number = number
        self.name = name
        self.lat = lat
        self.long = long
        self.flow = flow
        self.neighbors = neighbors
        
    def __eq__(self, other):
        return self.lat == other.lat and self.long == other.long
    
    def print(self):
        return({
            self.number: {
                'name': self.name,
                'lat': self.lat,
                'long': self.long,
            }
        })

def get_scats_dict(file_path):
    df = pd.read_csv(file_path, encoding='utf-8').fillna(0)
    scats_neighbours_dict = {}

    for _, row in df.iterrows():
        scats_number = str(row['SCATS Number']) # SCAT numbers needs to be universally strings for easy input and look up
        name = row['NAME']
        neighbors = row['NEIGHBOURS']
        lat = row['NB_LATITUDE']
        long = row['NB_LONGITUDE']
        neighbors_list = neighbors.split(';') if neighbors else []
        scats_neighbours_dict[scats_number] = Scat(scats_number, name, lat, long, 0, neighbors_list)

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

def process_data():
    # Read the dataset and fill empty values with 0
    df = pd.read_csv('data/myData.csv', encoding='utf-8').fillna(0)

    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')  # Convert date column to datetime
    df['Day Of Week'] = df['Date'].dt.dayofweek  # Extract the day of the week. Yes we have the Weeknum column, but we want to use a standardalized method (0-6, mon to sun)

    # Dropping non-relevant data
    df = df.drop(['SCATS Number', 'Location', 'CD_MELWAY', 'HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc', 'NB_TYPE_SURVEY', 'Date', 'Weeknum'], axis=1)

    # Transpose the 'V00' to 'V95' row and its value row into columns
    # We're gonna turn time into a feature/input as well
    df = df.melt(id_vars=['NB_LATITUDE', 'NB_LONGITUDE', 'Day Of Week'], var_name='Time', value_name='Flow')
    df['Time'] = df['Time'].str.replace('V', '').astype(int)

    # Normalize the time slots from 0 to 1 instead of 0 to 95 to avoid overfitting
    df['Time'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(df['Time'].values.reshape(-1, 1))

    # Standardize the traffic flow, but save the scaler.
    # When we make a prediction, the value that comes out will also be scaled,
    flow_scaler = StandardScaler() # So we'll need this later to "undo" the scaling
    df['Flow'] = flow_scaler.fit_transform(df['Flow'].values.reshape(-1, 1))

    # Standardize lat, long values to avoid overfitting
    lat_scaler = StandardScaler() # We also keep the scaler for lat and long so we can scale the input of new values
    df['NB_LATITUDE'] = lat_scaler.fit_transform(df['NB_LATITUDE'].values.reshape(-1, 1))

    long_scaler = StandardScaler()
    df['NB_LONGITUDE'] = long_scaler.fit_transform(df['NB_LONGITUDE'].values.reshape(-1, 1))

    # Reset the index, this newly modified dataframe is what we'll be working with
    df.reset_index(drop=True, inplace=True)

    # Get the input and target data
    x_train = df[['NB_LATITUDE', 'NB_LONGITUDE', 'Day Of Week', 'Time']].values
    y_train = df['Flow'].values

    # Reshape the input data for RNN
    x_train = x_train.reshape(x_train.shape[0], 4, 1)

    return x_train, y_train, flow_scaler, lat_scaler, long_scaler
