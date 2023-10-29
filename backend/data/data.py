import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

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

def get_lat_long_from_scats(data, scats):
    # Read data
    df = pd.read_csv(data, encoding='utf-8').fillna(0)

    # Filter the DataFrame based on the SCATS number. Makes a new dataframe thats just got the values for that SCATS number
    filtered_df = df[df['SCATS Number'] == int(scats)]

    # Get the lat and long from the new table
    lat = filtered_df['NB_LATITUDE'].iloc[0]
    long = filtered_df['NB_LONGITUDE'].iloc[0]

    return lat, long

def process_data(lags=12):
    # Read the dataset and fill empty values with 0
    df = pd.read_csv('data/myData.csv', encoding='utf-8').fillna(0)

    
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')  # Convert date column to datetime

    # Dropping non-relevant data
    df = df.drop(['SCATS Number', 'Location', 'CD_MELWAY', 'HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc', 'NB_TYPE_SURVEY', 'Weeknum'], axis=1)

    list_of_transformed_dfs = []
    grouped = df.groupby(['NB_LATITUDE', 'NB_LONGITUDE'])
    for _, group in grouped:
        list_of_transformed_dfs.append(transform_group(group, lags))

    #print(list_of_transformed_dfs)
    df = pd.concat(list_of_transformed_dfs).reset_index(drop=True)

    #Get the position of the first time column in the CSV
    column_position_of_time_column = df.columns.get_loc("V00")

    flow1 = df.to_numpy()[:, column_position_of_time_column:]

    flow_scaler = StandardScaler()
    # Select the columns to be scaled (from 'V00' to the last column)
    columns_to_scale = df.columns[df.columns.get_loc('V00'):]

    # # Fit and transform the scaler on the selected columns
    df[columns_to_scale] = flow_scaler.fit_transform(flow1)

    # Now, create a new MinMaxScaler for 'NB_LATITUDE'
    lat_scaler = StandardScaler()

    # Fit and transform the scaler on 'NB_LATITUDE'
    df['NB_LATITUDE'] = lat_scaler.fit_transform(df[['NB_LATITUDE']])

    # Now, create a new MinMaxScaler for 'NB_LATITUDE'
    long_scaler = StandardScaler()

    # Fit and transform the scaler on 'NB_LATITUDE'
    df['NB_LONGITUDE'] = long_scaler.fit_transform(df[['NB_LONGITUDE']])

    # Extract the day of the week from the 'Date' column
    df['Day'] = df['Date'].dt.day_name()

    # One-hot encode the day of the week
    days_encoded = pd.get_dummies(df['Day'], prefix='', prefix_sep='')
    days_encoded = days_encoded.astype(int)

    # Drop the original 'Date' and 'Day' columns and concatenate the new one-hot encoded columns
    df = df.drop(columns=['Date', 'Day'])
    df = pd.concat([df, days_encoded], axis=1)

    # Reorder columns to desired format
    ordered_cols = ['NB_LATITUDE', 'NB_LONGITUDE'] + days_encoded.columns.tolist() + [col for col in df if col.startswith('V')]
    df = df[ordered_cols]

    train = np.array(df)
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    return x_train, y_train, flow_scaler, lat_scaler, long_scaler

def transform_group(group, lags=3):
    """Transforms a single group of data with a variable number of VXX columns, handling a full month of dates."""
    values = group.iloc[:, 3:].values.flatten().tolist()
    
    # Since each location has entries for the entire month, 
    # we can safely append values from the start of the group to handle end-to-start shifting
    values += group.iloc[0:31, 3:].values.flatten().tolist()
    
    # Create a list to hold the transformed rows
    transformed_rows = []
    
    for i in range(len(group)*len(group.columns[3:])):
        # Extract the shifted values
        shifted_values = values[i:i+lags]
        
        # Only proceed if there are the required number of values to append
        if len(shifted_values) == lags:
            # Determine the date for the transformed row
            # Adjusted to take the date corresponding to the last V column value
            date_index = (i + lags - 1) // (len(group.columns) - 3)
            date = group['Date'].iloc[date_index % len(group)]
            
            # Append the transformed row
            transformed_rows.append([group['NB_LATITUDE'].iloc[0], 
                                     group['NB_LONGITUDE'].iloc[0], 
                                     date] + shifted_values)
    
    columns = ['NB_LATITUDE', 'NB_LONGITUDE', 'Date'] + ['V{:02}'.format(i) for i in range(lags)]
    return pd.DataFrame(transformed_rows, columns=columns)
