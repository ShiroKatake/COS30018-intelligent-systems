import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import csv
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

def process_data(lags=12):
    # Read the dataset and fill empty values with 0
    df = pd.read_csv('data/myData.csv', encoding='utf-8').fillna(0)

    
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')  # Convert date column to datetime
    #df['Day Of Week'] = df['Date'].dt.dayofweek  # Extract the day of the week. Yes we have the Weeknum column, but we want to use a standardalized method (0-6, mon to sun)

    # Dropping non-relevant data
    df = df.drop(['SCATS Number', 'Location', 'CD_MELWAY', 'HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc', 'NB_TYPE_SURVEY', 'Weeknum'], axis=1)

    list_of_transformed_dfs = []
    grouped = df.groupby(['NB_LATITUDE', 'NB_LONGITUDE'])
    for _, group in grouped:
        list_of_transformed_dfs.append(transform_group(group, lags))

    df = pd.concat(list_of_transformed_dfs).reset_index(drop=True)

    # Select the columns to be scaled (from 'V00' to the last column)
    columns_to_scale = df.columns[df.columns.get_loc('V00'):]

    flow_scaler = MinMaxScaler()
    # Fit and transform the scaler on the selected columns
    df[columns_to_scale] = flow_scaler.fit_transform(df[columns_to_scale])

    # Now, create a new MinMaxScaler for 'NB_LATITUDE'
    lat_scaler = MinMaxScaler()

    # Fit and transform the scaler on 'NB_LATITUDE'
    df['NB_LATITUDE'] = lat_scaler.fit_transform(df[['NB_LATITUDE']])

    # Now, create a new MinMaxScaler for 'NB_LATITUDE'
    long_scaler = MinMaxScaler()

    # Fit and transform the scaler on 'NB_LATITUDE'
    df['NB_LONGITUDE'] = long_scaler.fit_transform(df[['NB_LONGITUDE']])

    # 2. Extract the day of the week from the 'Date' column
    df['Day'] = df['Date'].dt.day_name()

    # 3. One-hot encode the day of the week
    days_encoded = pd.get_dummies(df['Day'], prefix='', prefix_sep='')
    days_encoded = days_encoded.astype(int)


    # 4. Drop the original 'Date' and 'Day' columns and concatenate the new one-hot encoded columns
    df = df.drop(columns=['Date', 'Day'])
    df = pd.concat([df, days_encoded], axis=1)

    # 5. Reorder columns to desired format
    ordered_cols = ['NB_LATITUDE', 'NB_LONGITUDE'] + days_encoded.columns.tolist() + [col for col in df if col.startswith('V')]
    df = df[ordered_cols]

    #print(f"TABLE:  \n\n {df}")



    train = np.array(df)
    np.random.shuffle(train)

    X_train = train[:, :-1]
    y_train = train[:, -1]

    return X_train, y_train, flow_scaler, lat_scaler, long_scaler









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
    
    # Adjust columns based on the number of V columns
    columns = ['NB_LATITUDE', 'NB_LONGITUDE', 'Date'] + ['V{:02}'.format(i) for i in range(lags)]

    return sort_transformed_data(pd.DataFrame(transformed_rows, columns=columns))

def sort_transformed_data(df, num_v_columns=3):
    """Sorts the transformed data based on the last V column."""
    last_v_column = 'V{:02}'.format(num_v_columns - 1)
    return df.sort_values(by=['NB_LATITUDE', 'NB_LONGITUDE', 'Date', last_v_column])