# SE-Europe-Data_Challenge

# Information about the data pipeline and model

**Data**
The training data is imported from 2022-01-01 to 2023-01-01. Of generation data, only reneweble energy generation data is imported.

The data is split to training and test sets, so that the test set is 442 data points long. However, this is not the asked 20% of the dataset because the requirements were clarified after the data processing part was already implemented according to previous understanding of instructions. 
The training set is split 80/20 to training and validation sets. The data is from ENTSO-E Transparency portal.

The following renewable energy sources are included in input data:
    - Biomass
    - Geothermal
    - Hydro_Pumped_Storage
    - Hydro_Run-of-river_and_poundage
    - Hydro_Water_Reservoir
    - Marine
    - Other_renewable
    - Solar
    - Wind_Offshore
    - Wind_Onshore
from Spain, UK, Germany, Denmark, Sweden, Hungary, Italy, Poland, and the Netherlands.
Also, the energy consumption from aforementioned countries is part of data.

The model input data has 75 columns.


**Pipeline evaluation**
The data ingestion pipeline is evaluated using the following metrics:
    - Data ingestion time: measures how long it took to perform data ingestion
    - Data processing time: measures how long it took to perform data processing and cleaning
    - CPU usage during data ingestion: measures the used CPU during the data ingestion, a resource utilization metric
    - CPU usage during data processing: measures the used CPU during the data processing and cleaning, a resource utilization metric
    - Memory usage during data ingestion: measures the used memory in megabytes during data ingestion, a resource utilization metric
    - Memory usage during data processing: measures the used memory in megabytes during data processing and cleaning, a resource utilization metric
    The following are reported for every csv (data metrics):
        - Absolute number of missing values (Null values)
        - Absolute number of missing rows
        - Percentage of missing rows: percentage of missing rows of all rows, a data completeness metric
        - Percentage of missing values: percentage of missing values of all values, a data completeness metric
        - Absolute number of duplicate rows

Results of data ingestion & processing pipeline metrics:
    - Data ingestion time: 416.73700881004333 seconds
    - CPU usage during data ingestion: 0.0%
    - Memory usage during data ingestion: 53.17578125 MB
    - Data processing time: 32.728888511657715 seconds
    - CPU usage during data processing: 0.0%
    - Memory usage dring data processing: 29.703125 MB
    - The data metrics are not reported here, but found from data_metrics.csv and during running the pipeline

According to these metrics data ingestion is relatively time-consuming and requires very little CPU, but a moderate amount of memory. Data processing, on the other hand, is much less time consuming almost less memory intensive.

Also, no data is lost during processing.

More detailed analysis of missing values:
Nearly all files only miss one row and this row is the one that needs to be added to the files with 1h time step because the files with smaller time step include data after 22.00 on the last day of 2022, while the ones with 1h time step do not. Some files have a large number of duplicate rows, but it needs to be taken into account that those files have smaller time step than the end file, so the absolute number of duplicates is not comparable to the absolute number of rows ('All rows').
The percentage of null values is always under 2%, which is quite good and looking at more fine-grained data, the missing values are often the area ids. There are some files with quite large number of missing rows (almost 90%), making the data from those somewhat unreliable because we cannot be sure whether the missing values are 0 or something else. However, mostly only 1 or 2 rows are missing with only 3 files that have a larger number of rows missing. One of them is the load of UK, which is the only load-file with many missing lines. The others are wind offshore for IT, wind onshore for UK and marine for SE.

**Model**
The used model is a GRU model using the Pytorch Gated Recurrent Unit (GRU) model and a linear classifier. Recurrent Neural Network (RNN) models are well suited for series-data and the GRU was chosen because it is less likely to overfit than for example a LSTM-model. This is an important consideration because the used dataset is relatively small.
Hyperparameters are tuned using grid search.

The best found hyperparameters are:
    - Number of layers: 2
    - Hidden size: 64


# Brief explanation of data cleaning and processing
The loaded csv-data files are converted into pandas dataframes and possible empty or NaN values are imputed by interpolation. Duplicate rows are removed and the time steps between consecutive start times and start and end times are all normalized to be 1 hour. The units are chacked, so that there are no data points with different unit values. No longer necessary columns are removed and the columns are renamed to communicate the country the values represent.

After these steps have been taken, the dataframes are combined together. Then, for every country, the generated energies are summed together and the load is sbtracted from this sums, giving the surplus green energy. The surplus green energies are compared and the label of the country with the largest surplus is chosen as the label. The labels are shifted by one, so that every row (one-hour timestep) has the country with the largest green energy surplus for the coming hour as the label.

# How to train this model?
The model can be trained with the following command: 
    python3 src/model_training.py

If you wish to also tune the hyperparameters in addition to training, use the --tuning_mode flag.

# How to run this pipeline?
The whole pipeline can be run with run_pipeline.sh. Remember to give  as arguments and have trained the model beforehand.

Use the following arguments:
    - start time: 2022-01-01
    - end time: 2023-01-01
    - raw data file: ./data
    - processed data file: data/processed_data.csv
    - model file: models/model.pkl
    - test data file: data/test.csv
    - predictions file: predictions/predictions.json
