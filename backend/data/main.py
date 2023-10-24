import pandas as pd

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

# Assuming you have your dataframe df read from 'data original.csv'
df = pd.read_csv('data original.csv')
# Group by latitude and longitude, transform and then concatenate the resulting DataFrames
list_of_transformed_dfs = []
grouped = df.groupby(['NB_LATITUDE', 'NB_LONGITUDE'])
for name, group in grouped:
    list_of_transformed_dfs.append(transform_group(group, 10))

df_transformed = pd.concat(list_of_transformed_dfs).reset_index(drop=True)

print(df_transformed)
