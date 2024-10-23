import pandas as pd

# Question 9:
def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
     # Load the dataset
    df = pd.read_csv(file_path)
    
    # Create a set of all unique locations (IDs)
    locations = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))
    
    # Initialize the distance matrix with infinity
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    
    # Set the diagonal to 0 (distance from a location to itself)
    np.fill_diagonal(distance_matrix.values, 0)

    # Populate the distance matrix with the direct distances
    for _, row in df.iterrows():
        from_location = row['id_start']
        to_location = row['id_end']
        distance = row['distance']
        
        # Fill in both directions to ensure symmetry
        distance_matrix.at[from_location, to_location] = distance
        distance_matrix.at[to_location, from_location] = distance

    # Use the Floyd-Warshall algorithm to find all-pairs shortest paths
    for k in locations:
        for i in locations:
            for j in locations:
                if distance_matrix.at[i, j] > distance_matrix.at[i, k] + distance_matrix.at[k, j]:
                    distance_matrix.at[i, j] = distance_matrix.at[i, k] + distance_matrix.at[k, j]
    
    return distance_matrix


# Example:
distance_matrix = calculate_distance_matrix("C:/Users/218882/Downloads/dataset-2.csv")




# Question 10:
def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Create a list to store the unrolled data
    unrolled_data = []

    # Iterate over the rows and columns of the distance matrix
    for id_start in df.index:
        for id_end in df.columns:
            if id_start != id_end:  # Exclude same id_start to id_end
                distance = df.at[id_start, id_end]
                unrolled_data.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    # Convert the list of dictionaries to a DataFrame
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df

# Example :
unrolled_df = unroll_distance_matrix(distance_matrix)
unrolled_df.head(10)




# Question 11:
def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
     # Calculate average distance for the reference ID
    ref_distances = df[df['id_start'] == reference_id]['distance']
    
    if ref_distances.empty:
        return pd.DataFrame(columns=['id_start', 'average_distance'])

    reference_avg_distance = ref_distances.mean()

    # Calculate the lower and upper thresholds
    lower_threshold = reference_avg_distance * 0.9
    upper_threshold = reference_avg_distance * 1.1

    # Group by id_start and calculate average distances
    average_distances = df.groupby('id_start')['distance'].mean().reset_index()

    # Filter IDs within the percentage threshold
    filtered_ids = average_distances[
        (average_distances['distance'] >= lower_threshold) &
        (average_distances['distance'] <= upper_threshold)
    ]

    # Sort by average distance
    filtered_ids = filtered_ids.sort_values(by='distance')

    return filtered_ids

# Example:
result_df = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
print(result_df)





# Question 12:
def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type and add new columns to the DataFrame
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate

    return df

# Example usage:
# unrolled_df = ...  # This would be the result from the unroll_distance_matrix function
toll_rates_df = calculate_toll_rate(unrolled_df)
print(toll_rates_df.head(10))





# Question 13:
def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Define the days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Create lists to hold new values
    start_days = []
    start_times = []
    end_days = []
    end_times = []
    
    # Loop through each unique (id_start, id_end) pair
    for (id_start, id_end), group in df.groupby(['id_start', 'id_end']):
        for day in days_of_week:
            # For each day, we will create a 24-hour span
            for hour in range(24):
                start_time_val = time(hour, 0)
                end_time_val = time(hour, 59, 59)  # Represents the end of that hour
                
                # Calculate toll rates based on time
                if day in ['Saturday', 'Sunday']:
                    # Apply a constant discount factor of 0.7
                    discount_factor = 0.7
                else:  # Weekdays
                    if hour < 10:
                        discount_factor = 0.8  # 00:00 - 10:00
                    elif 10 <= hour < 18:
                        discount_factor = 1.2  # 10:00 - 18:00
                    else:
                        discount_factor = 0.8  # 18:00 - 23:59

                # Calculate toll rates for each vehicle type using the discount factor
                for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']:
                    toll_col = vehicle
                    toll_rate = group[toll_col].mean() * discount_factor
                    # Update the vehicle toll rate in the DataFrame
                    df.loc[(df['id_start'] == id_start) & (df['id_end'] == id_end), toll_col] = toll_rate

                # Append the values for new columns
                start_days.append(day)
                start_times.append(start_time_val)
                end_days.append(day)
                end_times.append(end_time_val)

    # Add the new columns to the DataFrame
    df['start_day'] = start_days
    df['start_time'] = start_times
    df['end_day'] = end_days
    df['end_time'] = end_times

    return df

# Example:
time_based_toll_rates_df = calculate_time_based_toll_rates(toll_rates_df)
print(time_based_toll_rates_df)

    return df
