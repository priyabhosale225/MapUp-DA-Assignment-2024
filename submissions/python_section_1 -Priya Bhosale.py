from typing import Dict, List

import pandas as pd
import numpy as np 


# Question 1:
def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    length = len(lst)
    
    for i in range(0, length, n):
        group = lst[i:i + n]
        reversed_group = []
        
        # Reverse the current group manually
        for j in range(len(group)):
            reversed_group.append(group[len(group) - 1 - j])
        
        result.extend(reversed_group)
    
    return result
    



# Question 2:
def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}
    
    for string in lst:
        length = len(string)
        if length not in length_dict:
            length_dict[length] = []
        length_dict[length].append(string)
    
    # Sort the dictionary by key (length)
    sorted_length_dict = dict(sorted(length_dict.items()))
    
    return sorted_length_dict
    



# Question 3:
def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    flat_dict = {}

    def flatten(current_dict: Dict, parent_key: str = ''):
        for key, value in current_dict.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                flatten(value, new_key)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        flatten(item, f"{new_key}[{index}]")
                    else:
                        flat_dict[f"{new_key}[{index}]"] = item
            else:
                flat_dict[new_key] = value

    flatten(nested_dict)
    return flat_dict
    




# Question 4:
def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])  # Append a copy of the current permutation
            return
        
        seen = set()  # To track elements we've used at this level
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue  # Skip duplicates
            seen.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start]  # Swap
            backtrack(start + 1)  # Recurse
            nums[start], nums[i] = nums[i], nums[start]  # Swap back (backtrack)

    nums.sort()  # Sort to facilitate duplicate handling
    result = []
    backtrack(0)
    return result





# Question 5: 
import re 

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    # Regular expression pattern for matching the date formats
    date_pattern = r'''
        \b                     # Word boundary
        (                     # Start of the main pattern group
            \d{2}-\d{2}-\d{4} |    # dd-mm-yyyy
            \d{2}/\d{2}/\d{4} |    # mm/dd/yyyy
            \d{4}\.\d{2}\.\d{2}    # yyyy.mm.dd
        )                     # End of the main pattern group
        \b                     # Word boundary
    '''
    
    # Find all matches in the input text
    matches = re.findall(date_pattern, text, re.VERBOSE)
    
    return matches





# Question 6: 
import polyline

def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points
    on the Earth specified in decimal degrees (latitude and longitude).
    
    Args:
        coord1: A tuple containing (latitude, longitude) of the first point.
        coord2: A tuple containing (latitude, longitude) of the second point.
        
    Returns:
        Distance in meters.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Radius of Earth in meters (mean radius)
    r = 6371000  
    return r * c


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude,
    and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Decode the polyline
    decoded_coords = polyline.decode(polyline_str)

    # Prepare lists for DataFrame
    latitudes = []
    longitudes = []
    distances = [0]  # Distance for the first point is 0

    # Calculate distances
    for i, (lat, lon) in enumerate(decoded_coords):
        latitudes.append(lat)
        longitudes.append(lon)
        if i > 0:
            distance = haversine(decoded_coords[i - 1], (lat, lon))
            distances.append(distance)

    # Create DataFrame
    df = pd.DataFrame({
        'latitude': latitudes,
        'longitude': longitudes,
        'distance': distances
    })
    
    return df





# Question 7:
def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    
    # Rotate the matrix by 90 degrees clockwise
    rotated_matrix = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]

    # Create the final matrix with sums
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i])  # Sum of the i-th row
            col_sum = sum(rotated_matrix[k][j] for k in range(n))  # Sum of the j-th column
            final_matrix[i][j] = row_sum + col_sum - rotated_matrix[i][j]  # Exclude itself
    
    return final_matrix





# Question 8:
def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
   
    # Ensure startDay and endDay are strings, and combine them with their respective times
    df['start_timestamp'] = pd.to_datetime(df['startDay'].astype(str) + ' ' + df['startTime'], errors='coerce')
    df['end_timestamp'] = pd.to_datetime(df['endDay'].astype(str) + ' ' + df['endTime'], errors='coerce')

    # Drop rows where timestamp conversion failed
    df = df.dropna(subset=['start_timestamp', 'end_timestamp'])

    # Initialize a list to store results
    results = []

    # Group by (id, id_2)
    for (id_val, id_2_val), group in df.groupby(['id', 'id_2']):
        # Get the unique days and the min/max timestamps for the group
        unique_days = group['start_timestamp'].dt.day_name().unique()
        min_time = group['start_timestamp'].min()
        max_time = group['end_timestamp'].max()

        # Check if there are exactly 7 unique days
        days_complete = len(unique_days) == 7
        
        # Check if the time covers the full 24 hours (from 00:00 to 23:59)
        time_complete = (min_time.time() <= pd.Timestamp("00:00:00").time() and
                         max_time.time() >= pd.Timestamp("23:59:59").time())

        # Append the result
        results.append(((id_val, id_2_val), not (days_complete and time_complete)))

    # Convert results into a Series with MultiIndex
    results_series = pd.Series(dict(results))
    results_series.index = pd.MultiIndex.from_tuples(results_series.index, names=['id', 'id_2'])

    return results_series

    return pd.Series()


