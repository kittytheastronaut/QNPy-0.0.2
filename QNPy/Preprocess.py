import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
from math import ceil
import os
import glob 
import warnings
import dill

import csv


def clean_and_save_outliers(input_folder, output_folder, threshold=3.0):
    """
    Clean data in CSV files within the input folder, remove outliers, and save cleaned files in the output folder.

    Parameters:
        input_folder (str): Path to the folder containing input CSV files.
        output_folder (str): Path to the folder where cleaned CSV files will be saved.
        threshold (float, optional): Threshold for outlier detection in terms of standard deviations. Default is 3.0.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through each CSV file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)

            # Load the data from the CSV file
            data = []
            with open(input_filepath, 'r') as infile:
                header = infile.readline().strip().split(',')  # Assuming header is comma-separated
                for line in infile:
                    line_data = line.strip().split(',')  # Assuming data is comma-separated
                    data.append([float(val) for val in line_data])

            data = np.array(data)

            # Extract columns
            mjd_index = header.index('mjd')
            mag_index = header.index('mag')
            magerr_index = header.index('magerr')

            mjd, mag, magerr = data[:, mjd_index], data[:, mag_index], data[:, magerr_index]

            # Compute the Z-Score for each point in the light curve
            z_scores = np.abs((mag - np.mean(mag)) / np.std(mag))

            # Find the indices of the non-outlier points.
            good_indices = np.where(z_scores <= threshold)[0]

            # Create new arrays with only the non-outlier points
            clean_mjd = mjd[good_indices]
            clean_mag = mag[good_indices]
            clean_magerr = magerr[good_indices]

            # Save the cleaned data to a new CSV file
            with open(output_filepath, 'w', newline='') as outfile:
                writer = csv.writer(outfile, delimiter=',')  # Use comma as delimiter
                writer.writerow(header)  # Write header row
                for t, f, e in zip(clean_mjd, clean_mag, clean_magerr):
                    writer.writerow([t, f, e])  # Write data row

            print(f"Cleaned and saved {filename} to {output_filepath}")


def clean_save_aggregate_data(input_folder, output_folder, threshold_aggregation=5, threshold_outliers=3.0):
    """
    Clean data in CSV files within the input folder, remove outliers, aggregate time and fluxes,
    and save cleaned files in the output folder.

    Parameters:
        input_folder (str): Path to the folder containing input CSV files.
        output_folder (str): Path to the folder where cleaned CSV files will be saved.
        threshold_aggregation (float, optional): Threshold for time aggregation. Default is 5.
        threshold_outliers (float, optional): Threshold for outlier detection in terms of standard deviations. Default is 3.0.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define header names
    header = ["mjd", "mag", "magerr"]

    # Loop through each CSV file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)

            # Load the data from the CSV file with comma as the delimiter
            data = np.loadtxt(input_filepath, delimiter=',', skiprows=1)  # Assuming the first row is header

            # Extract columns
            times = data[:, 0]
            magnitudes = data[:, 1]
            magerr = data[:, 2]  # Assuming magerr is in the third column

            # Remove outliers
            z_scores = np.abs((magnitudes - np.mean(magnitudes)) / np.std(magnitudes))
            good_indices = np.where(z_scores <= threshold_outliers)[0]
            clean_times = times[good_indices]
            clean_magnitudes = magnitudes[good_indices]

            # Aggregate times and fluxes
            sorted_indices = np.argsort(clean_times)
            clean_times = clean_times[sorted_indices]
            clean_magnitudes = clean_magnitudes[sorted_indices]
            dff = np.min(np.diff(clean_times))
            aggregated_times = []
            aggregated_magnitudes = []

            i = 0
            while i < len(clean_times):
                current_time_sum = clean_times[i]
                current_magnitude_sum = clean_magnitudes[i]

                n_aggregated = 1  # At least one time instance will be aggregated (itself)

                # Look forward to see how many time instances can be aggregated
                j = i + 1
                while j < len(clean_times) and (clean_times[j] - clean_times[j-1]) < threshold_aggregation * dff:
                    current_magnitude_sum += clean_magnitudes[j]
                    current_time_sum += clean_times[j]
                    n_aggregated += 1
                    j += 1

                if n_aggregated > 5:
                    aggregated_times.append(current_time_sum / n_aggregated)
                    aggregated_magnitudes.append(current_magnitude_sum / n_aggregated)  # Weighted average of the magnitudes
                    i = j
                else:
                    aggregated_times.append(clean_times[i])
                    aggregated_magnitudes.append(clean_magnitudes[i])
                    i += 1

            # Save the cleaned data to a new CSV file without delimiter
            with open(output_filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)  # Write header row
                for t, f, e in zip(aggregated_times, aggregated_magnitudes, magerr):
                    writer.writerow([t, f, e])  # Write data row

            print(f"Cleaned and saved {filename} to {output_filepath}")

    # Print a message indicating the end of processing
    print("Processing completed.")


def backward_pad_curves(folder_path, output_folder, desired_observations=100):
    """Backward padding the light curves with the last observed value for mag and magerr.
    If your data contains 'time' values it'll add +1 for padded values,
    and if your data contains 'MJD' values it will add +0.2

    ARGS:
    :param str folder_path: The path to a folder containing the .csv files.
    :param str output_path: The path to a folder for saving the padded lc.
    :param int desired_observations: The number of points that our package is demanding is 100 but it can be more.

    :return: The padded light curves.
    :rtype: object
    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of CSV files in the input folder
    csv_files = [filename for filename in os.listdir(folder_path) if filename.endswith('.csv')]

    # Initialize variables to store maximum number of rows and average values per curve
    max_rows = 0
    average_mag_dict = {}
    average_magerr_dict = {}
    first_column_header = None

    # Iterate over the CSV files to find the maximum number of rows and calculate averages per curve
    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)
        try:
            # Read the CSV file and load the data into a pandas DataFrame
            data = pd.read_csv(file_path)

            # Get the number of rows in the DataFrame
            num_rows = data.shape[0]

            # Determine the header of the first column
            first_column_header = data.columns[0]

            # Assume the second and third columns are mag and magerr
            # Calculate average mag and magerr per curve
            average_mag_dict[filename] = data.iloc[:, 1].mean()
            average_magerr_dict[filename] = data.iloc[:, 2].mean()

            # Update the maximum row count
            if num_rows > max_rows:
                max_rows = num_rows

        except pd.errors.EmptyDataError:
            print(f"Error: Empty file encountered: {filename}")

    # Create new DataFrames with backward padding and ensure a minimum of desired_observations
    new_data_dict = {}

    # Iterate over the CSV files
    for filename in csv_files:
        try:
            # Read the CSV file and load the data into a pandas DataFrame
            data = pd.read_csv(os.path.join(folder_path, filename))

            # Calculate the number of missing rows to reach desired_observations
            missing_rows = desired_observations - len(data)

            # Check the header of the first column and add the appropriate values
            if first_column_header.lower() == 'mjd':
                data['mjd'] = data['mjd'].astype(float)  # Ensure "mjd" column is numeric
                mjd_increment = 0.2  # Specify the MJD increment here
                last_mjd = data.iloc[-1, 0]
                extra_data = pd.DataFrame({
                    first_column_header: np.arange(last_mjd + mjd_increment, last_mjd + mjd_increment * (missing_rows + 1), mjd_increment),
                    'mag': data.iloc[-1, 1],     # Backward-fill the last available mag value
                    'magerr': data.iloc[-1, 2]   # Backward-fill the last available magerr value
                })
            else:
                last_value = data.iloc[-1, 0]
                extra_data = pd.DataFrame({
                    first_column_header: np.arange(last_value + 1, last_value + 1 + missing_rows),
                    'mag': data.iloc[-1, 1],     # Backward-fill the last available mag value
                    'magerr': data.iloc[-1, 2]   # Backward-fill the last available magerr value
                })

            data = pd.concat([data, extra_data], ignore_index=True)

            # Pad to exactly 100 points if the longest curve has fewer than 100 points
            if max_rows < desired_observations:
                pad_rows = desired_observations - len(data)
                if first_column_header.lower() == 'mjd':
                    mjd_increment = 0.2
                    extra_data = pd.DataFrame({
                        first_column_header: np.arange(data.iloc[-1, 0] + mjd_increment, data.iloc[-1, 0] + mjd_increment * (pad_rows + 1), mjd_increment),
                        'mag': average_mag_dict[filename],
                        'magerr': average_magerr_dict[filename]
                    })
                else:
                    extra_data = pd.DataFrame({
                        first_column_header: np.arange(data.iloc[-1, 0] + 1, data.iloc[-1, 0] + 1 + pad_rows),
                        'mag': average_mag_dict[filename],
                        'magerr': average_magerr_dict[filename]
                    })
                data = pd.concat([data, extra_data], ignore_index=True)

            # Save the new DataFrame to a new CSV file in the output folder with the original filename
            output_file = os.path.join(output_folder, filename)
            data.to_csv(output_file, index=False)

            print(f"Created new file: {output_file}")
        except pd.errors.EmptyDataError:
            print(f"Error: Empty file encountered: {filename}")


def transform(data):
    """Transforming data into [-2,2]x[-2,2] range. This function needs to be uploaded before using it.

    Args:
    :param data: Your data must contain: MJD or time, mag-magnitude, and magerr-magnitude error.
    :type data: object
    """
    x_data = np.array(data['mjd'])
    y_data = np.array(data['mag'])
    z_data = np.array(data['magerr'])
    
    # If magerr is 0 replace it with 0.01*mag
    for i in range(len(z_data)):
        if z_data[i] == 0:
            z_data[i] = 0.01 * y_data[i]
    
    Mmean = np.mean(y_data)
    
    ### mapping time and flux to interval [-2,2]
    a = -2
    b = 2

    Ax = min(x_data)
    Bx = max(x_data)

    Ay = min(y_data)
    By = max(y_data)
   
    # Make series with added noise
    y_dataeplus = y_data + z_data
    # Make series with subtracted noise
    y_dataeminus = y_data - z_data

    Ayeplus = min(y_dataeplus)
    Byeplus = max(y_dataeplus)

    Ayeminus = min(y_dataeminus)
    Byeminus = max(y_dataeminus)

    x_data = (x_data - Ax) * (b-a) / (Bx - Ax) + a
    y_data = (y_data - Ay) * (b-a) / (By - Ay) + a
    
    y_dataeplus1 = (y_dataeplus - Ayeplus) * (b-a) / (Byeplus - Ayeplus) + a
    y_dataeminus1 = (y_dataeminus - Ayeminus) * (b-a) / (Byeminus - Ayeminus) + a

    data_result = {'time': x_data,
                   'cont': y_data,
                   'conterr': z_data}
    data_result = pd.DataFrame(data_result)

    dataeplus_result = {'time': x_data,
                        'cont': y_dataeplus1,
                        'conterr': z_data}
    dataeplus_result = pd.DataFrame(dataeplus_result)

    dataeminus_result = {'time': x_data,
                         'cont': y_dataeminus1,
                         'conterr': z_data}
    dataeminus_result = pd.DataFrame(dataeminus_result)

    return data_result, dataeplus_result, dataeminus_result, Ax, Bx, Ay, By


import os
import warnings
import dill

def transform_and_save(files, data_src, data_dst, transform):
    """Transforms and saves a list of CSV files. The function also saves tr coefficients as a pickle file named trcoeff.pickle.

    Args:
    :param list files: A list of CSV or TXT file names.
    :param str DATA_SRC: The path to the folder containing the CSV or TXT files.
    :param str DATA_DST: The path to the folder where the transformed CSV or TXT files will be saved.
    :param function transform: The transformation function defined previously.

    :return: A list of transformation coefficients for each file, where each element is a list containing the file name and the transformation coefficients Ax, Bx, Ay, and By.
    :rtype: list
    """
    number_of_points = []
    counter = 0
    trcoeff = []

    for file in files:
        lcName = file.split(".")[0]
        tmpDataFrame = pd.read_csv(os.path.join(data_src, file))
        
        # Catch runtime warnings in transform function
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                data_result, dataeplus_result, dataeminus_result, Ax, Bx, Ay, By = transform(tmpDataFrame)
            except Warning as e:
                print('warning found:', e)
                print(lcName)
                continue
            except Exception as e:
                print('error found:', e)
                print(lcName)
                continue
        
        if data_result is not None:
            counter += 1
            number_of_points.append(len(data_result.index))
            
            # Save the original data
            filename = os.path.join(data_dst, lcName + '_original.csv')
            data_result.to_csv(filename, index=False)
            
            # Save the plus light curves
            filename_plus = os.path.join(data_dst, lcName + '_plus.csv')
            dataeplus_result.to_csv(filename_plus, index=False)
            
            # Save the minus light curves
            filename_minus = os.path.join(data_dst, lcName + '_minus.csv')
            dataeminus_result.to_csv(filename_minus, index=False)
            
            trcoeff.append([lcName, Ax, Bx, Ay, By])
        
    dill.dump(trcoeff, file=open("trcoeff.pickle", "wb"))
    return number_of_points, trcoeff



