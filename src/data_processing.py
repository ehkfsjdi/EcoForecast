import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import time
from utils import measure_CPU_and_memory_usage
import glob
from math import ceil

def load_data(file_path):
    # Load data from CSV file
    df = pd.read_csv(file_path)

    return df

def clean_data(df, verbose=False):
    '''Handles missing values and duplicate rows'''
    # Number of missing values
    missing_values = (df.isnull().sum().sum(), (df.isnull().sum().sum()/df.count().sum()))
    if verbose:
        # Print more infor about missing values
        print('Missing values:', df.isnull().sum())
    # Handle missing values by linear interpolation
    # Get all numeric value columns
    cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in cols:
        df[col].interpolate(method='linear', limit_direction='both', inplace=True)
    # Check for duplicates
    duplicateRows = df[df.duplicated(subset='StartTime')].count().sum()
    # Drop duplicates based on 'StartTime' while keeping the first occurrence
    df_clean = df.drop_duplicates(subset='StartTime', keep='first')
    # Reset the index if needed
    df_clean.reset_index(drop=True, inplace=True)

    return df_clean, missing_values, duplicateRows
    
def clear_times(df, step=1, start_t=pd.to_datetime('2021-12-31T23:00+00:00Z', format="%Y-%m-%dT%H:%M+%S:%fZ"), end_t=pd.to_datetime('2022-12-31T23:00+00:00Z', format="%Y-%m-%dT%H:%M+%S:%fZ")):
    '''Clears the start and end time columns'''
    empty_df = pd.DataFrame()
    # Make StartTime and EndTime columns given absolute start time and absolute end time and step
    empty_df['StartTime'] = pd.date_range(start=start_t, end=end_t, freq=f'{60*step}T')
    empty_df['EndTime'] = empty_df['StartTime'] + pd.Timedelta(hours=step)
    # Merge start times, end times and the dataframe
    new_df = empty_df.merge(df, how='outer')
    # Count number of missing rows
    missing_rows = ceil((len(new_df)-len(df))/(1/step))
    # Replace generated NaN values with 0
    for col in new_df.columns:
        if col == 'UnitName':
            new_df[col].fillna('MAW', inplace=True)
            continue
        new_df[col].fillna(0, inplace=True)

    return new_df, missing_rows

def sum_timesteps(df, step, label):
    n = 1/step
    # Sum together every n time steps
    aggregated = df[label].groupby(df.index // n * n).apply(lambda x: x.iloc[:].sum())
    df = df.loc[aggregated.index]
    df[label] = aggregated
    # Change the end times
    df['EndTime'] = df['StartTime'] + pd.Timedelta(hours=1)
    df = df.reset_index(drop=True)

    return df

def check_time_duration(name, load=False):
    '''Check that the times have differences of 1h and all start and end at the same time'''
    df = load_data(name)
    label = 'quantity'
    if load:
        label = 'Load'
    # Check that all the time steps are as long
    df['EndTime'] = pd.to_datetime(df['EndTime'], format="%Y-%m-%dT%H:%M+%S:%fZ")
    df['StartTime'] = pd.to_datetime(df['StartTime'], format="%Y-%m-%dT%H:%M+%S:%fZ")
    duration = df['EndTime'] - df['StartTime']

    # Clean the data
    df, missing_vals, duplicates = clean_data(df)
    missing_rows = 0

    # If there are multiple time step types, dividethe df according to them and process separately
    if len(duration.unique()) > 1:
        df_new = pd.DataFrame()
        for i in duration.unique():
            # Get the slice of df that has this time step type
            df_part = df.loc[duration == i]
            step = i/pd.Timedelta(hours=1)
            # use an appropriate 1st start time and end time
            # If we are dealing with the first part
            if df_part.index[0] == 0:
                end_t = df_part['StartTime'].dt.ceil('H').iloc[len(df_part)-1]  # round up to ensure we don't lose data
                df_part, missing_r = clear_times(df_part, step, end_t=end_t)
            # If we are dealing with the last part
            elif df_part.index[-1] == (len(df)-1):
                start_t = df_part['StartTime'].dt.floor('H').iloc[0]  # round down to ensure we don't lose data
                df_part, missing_r = clear_times(df_part, step, start_t=start_t)
            # If we have a middle part
            else:
                start_t = df_part['StartTime'].dt.floor('H').iloc[0]  # round down to ensure we don't lose data
                end_t = df_part['StartTime'].dt.ceil('H').iloc[len(df_part)-1]  # round up to ensure we don't lose data
                df_part, missing_r = clear_times(df_part, step, start_t=start_t, end_t=end_t)
            
            # Combine 15 and 30 min time steps
            if step != 1:
                df_part = sum_timesteps(df_part, step, label)

            if df_new.empty:
                df_new = df_part
                continue
            # Concatenate the partial dfs together
            df = pd.concat((df_new, df_part))
            missing_rows += missing_r

        # Sum all rows with the same start time
        # define how to aggregate various fields
        agg_functions = {'UnitName': 'first', label: 'sum'}
        df = df.groupby(['StartTime', 'EndTime']).aggregate(agg_functions).reset_index()

        return df, missing_vals, duplicates, missing_rows

    step = duration.unique()[0]/pd.Timedelta(hours=1)
    df, missing_rows = clear_times(df, step=step)
    df = df[['StartTime', 'EndTime', 'UnitName', label]]

    # Combine 15 and 30 min time steps
    if step != 1:
        df = sum_timesteps(df, step, label)

    return df, missing_vals, duplicates, missing_rows

def check_units(df, unit):
    # Check that the units are expected
    if not df.loc[df['UnitName'] != unit].empty:
        raise Exception('Unknown unit')
    else:
        return True

def rename_columns(df, name, load=False):
    # Rename columns of a df depending on name
    if load is True:
        split_name = name.split('_')[-1].split('.')
        new_name = f'{split_name[0]}_Load'
        df = df.rename(columns={"Load": new_name})
        df = df.drop(columns=['UnitName'])
    else:
        split_name = name.split('_')
        prefix = '_'.join(split_name[2:]).split('.')[0]
        new_name = f'{prefix}_{split_name[1]}'
        df = df.rename(columns={"quantity": new_name})
        df = df.drop(columns=['UnitName'])
    return df

def combine_csvs(files, load=False):
    df = pd.DataFrame()
    metrics = pd.DataFrame(columns=['Null values', 'Missing rows', 'All rows', 'Missing rows %',
        'Null values %', 'Duplicate rows'])
    # Merge all imported data into raw data file
    for file in files:
        # Check the time intervals and units to enable merging dataframes
        intermediate, nulls, duplicates, missing_rows = check_time_duration(file, load)
        check_units(intermediate, 'MAW')
        # Read csvs and rename columns
        renamed = rename_columns(intermediate, file, load)
        # Add a row to the metrics
        metrics.loc[len(metrics)] = {'Null values': nulls[0], 'Missing rows': missing_rows, 
            'All rows': len(renamed), 'Missing rows %': missing_rows/len(renamed), 
            'Null values %': nulls[1], 'Duplicate rows': duplicates}
        # Merge to dataframe
        if df.empty:
            df = renamed
            continue
        df = df.merge(renamed, how='outer')

    return df, metrics

def process_raw_data_files(path='./data'):
    # Process raw csv files to right format
    # Load csv files
    load_files = glob.glob(path + "/load_*.csv")
    # Gen csv files
    gen_files = glob.glob(path + "/gen_*.csv")
    # Process the csvs into one csv
    load_df, load_m = combine_csvs(load_files, True)
    gen_df, gen_m = combine_csvs(gen_files)
    # Merge the dataframes
    df = load_df.merge(gen_df, how='outer')
    metrics = pd.concat((load_m, gen_m)).reset_index(drop=True)
    print('Data cleaning metrics: ', metrics)
    # Save the metrics into a file
    save_data(metrics, './data/data_metrics.csv')

    return df

def preprocess_data(df):
    # Generate new features, transform existing features, resampling, etc.
    df_processed = df.copy()
    country_ids = {
        'SP': 0, # Spain
        'UK': 1, # United Kingdom
        'DE': 2, # Germany
        'DK': 3, # Denmark
        'HU': 5, # Hungary
        'SE': 4, # Sweden
        'IT': 6, # Italy
        'PO': 7, # Poland
        'NE': 8 # Netherlands
    }
    # Ensure no NA values remain
    df_processed = df_processed.fillna('0')
    for key in country_ids.keys():
        # Sum together all the renewable generated energies per country
        df_processed[f'green_energy_{key}'] = df.filter(regex=f'_{key}').sum(axis=1)
        # Create a new column for renewable energy surplus
        df_processed[f'surplus_{key}'] = df_processed[f'green_energy_{key}'] - df_processed[f'{key}_Load']
        # Shift the surplus column backward, so that every line has the surplus of the next hour
        df_processed[f'surplus_{key}'] = df_processed[f'surplus_{key}'].shift(periods=-1)

    # Find the country with the biggest energy surplus
    df_processed['label'] = df_processed.filter(regex=('surplus')).idxmax(axis=1)
    # Now transform the labels into the country IDs
    df_processed['label'][:-1] = df_processed['label'][:-1].apply(lambda x: country_ids[x.split('_')[-1]])
    # Drop the surplus columns
    df_processed = df_processed.drop(columns=df_processed.filter(regex=('surplus')).columns)

    # Cahnge datetime columns to numeric
    df_processed['StartTime'] = df_processed['StartTime'].apply(lambda x: int(x.timestamp()))
    df_processed['EndTime'] = df_processed['EndTime'].apply(lambda x: int(x.timestamp()))

    return df_processed

def save_data(df, output_file):
    # Save processed data to a CSV file
    df.to_csv(output_file, index=False)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='./data',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    # Combine created csv files and clean them
    df_clean = process_raw_data_files(path=input_file)
    # Preprocess the data
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)

if __name__ == "__main__":
    # Measure CPU and memory usage and time at the start of data processing
    start_cpu, start_memory = measure_CPU_and_memory_usage()
    # Start time
    start_time = time.time()

    args = parse_arguments()
    main(args.input_file, args.output_file)

    # Calculate used CPU and memory
    end_cpu, end_memory = measure_CPU_and_memory_usage()
    cpu_usage = end_cpu - start_cpu
    memory_usage = end_memory - start_memory
    # End time
    end_time = time.time()
    # Calculate the data processing time
    processing_time = end_time - start_time

    print(f"Data processing time: {processing_time} seconds")
    print(f"CPU usage: {cpu_usage}%")
    print(f"Memory usage: {memory_usage} MB")
