import argparse
import datetime
import pandas as pd
from utils import perform_get_request, xml_to_load_dataframe, xml_to_gen_data, measure_CPU_and_memory_usage
import time


def get_load_data_from_entsoe(regions, periodStart='202302240000', periodEnd='202303240000', output_path='./data'):
    
    # TODO: There is a period range limit of 1 year for this API. Process in 1 year chunks if needed
    
    # URL of the RESTful API
    url = 'https://web-api.tp.entsoe.eu/api'

    # General parameters for the API
    # Refer to https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html#_documenttype
    params = {
        'securityToken': '1d9cd4bd-f8aa-476c-8cc1-3442dc91506d',
        'documentType': 'A65',
        'processType': 'A16',
        'outBiddingZone_Domain': 'FILL_IN', # used for Load data
        'periodStart': periodStart, # in the format YYYYMMDDHHMM
        'periodEnd': periodEnd # in the format YYYYMMDDHHMM
    }

    # Loop through the regions and get data for each region
    for region, area_code in regions.items():
        print(f'Fetching data for {region}...')
        params['outBiddingZone_Domain'] = area_code
    
        # Use the requests library to get data from the API for the specified time range
        response_content = perform_get_request(url, params)

        # Response content is a string of XML data
        df = xml_to_load_dataframe(response_content)

        # Save the DataFrame to a CSV file
        df.to_csv(f'{output_path}/load_{region}.csv', index=False)
       
    return

def get_gen_data_from_entsoe(regions, periodStart='202302240000', periodEnd='202303240000', output_path='./data'):
    
    # TODO: There is a period range limit of 1 day for this API. Process in 1 day chunks if needed
    # The renewable energy types and codes
    renewable_psr_types = {
        'B01': 'Biomass', 'B09': 'Geothermal', 'B10':'Hydro_Pumped_Storage', 
        'B11': 'Hydro_Run-of-river_and_poundage', 'B12': 'Hydro_Water_Reservoir', 'B13': 'Marine', 
        'B15': 'Other_renewable', 'B16': 'Solar', 'B18': 'Wind_Offshore', 'B19': 'Wind_Onshore'
    }

    # URL of the RESTful API
    url = 'https://web-api.tp.entsoe.eu/api'

    # General parameters for the API
    params = {
        'securityToken': '1d9cd4bd-f8aa-476c-8cc1-3442dc91506d',
        'documentType': 'A75',
        'processType': 'A16',
        'outBiddingZone_Domain': 'FILL_IN', # used for Load data
        'in_Domain': 'FILL_IN', # used for Generation data
        'periodStart': periodStart, # in the format YYYYMMDDHHMM
        'periodEnd': periodEnd # in the format YYYYMMDDHHMM
    }

    # Loop through the regions and get data for each region
    for region, area_code in regions.items():
        print(f'Fetching data for {region}...')
        params['outBiddingZone_Domain'] = area_code
        params['in_Domain'] = area_code
    
        # Only fetch the data of renewable energy
        for psr_type in renewable_psr_types.keys():
            params['PsrType'] = psr_type
            # Use the requests library to get data from the API for the specified time range
            response_content = perform_get_request(url, params)

            # Response content is a string of XML data
            dfs = xml_to_gen_data(response_content)

            # Save the dfs to CSV files
            for psr_type, df in dfs.items():
                # Save the DataFrame to a CSV file
                df.to_csv(f'{output_path}/gen_{region}_{renewable_psr_types[psr_type]}.csv', index=False)
    
    return

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data ingestion script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--start_time', 
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), 
        default=datetime.datetime(2023, 1, 1), 
        help='Start time for the data to download, format: YYYY-MM-DD'
    )
    parser.add_argument(
        '--end_time', 
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), 
        default=datetime.datetime(2023, 1, 2), 
        help='End time for the data to download, format: YYYY-MM-DD'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='./data',
        help='Name of the output file'
    )
    return parser.parse_args()

def main(start_time, end_time, output_path):
    
    regions = {
        'HU': '10YHU-MAVIR----U',
        'IT': '10YIT-GRTN-----B',
        'PO': '10YPL-AREA-----S',
        'SP': '10YES-REE------0',
        'UK': '10Y1001A1001A92E',
        'DE': '10Y1001A1001A83F',
        'DK': '10Y1001A1001A65H',
        'SE': '10YSE-1--------K',
        'NE': '10YNL----------L',
    }

    # Transform start_time and end_time to the format required by the API: YYYYMMDDHHMM
    start_time = start_time.strftime('%Y%m%d%H%M')
    end_time = end_time.strftime('%Y%m%d%H%M')

    # Get Load data from ENTSO-E
    get_load_data_from_entsoe(regions, start_time, end_time, output_path)

    # Get Generation data from ENTSO-E
    get_gen_data_from_entsoe(regions, start_time, end_time, output_path)

if __name__ == "__main__":
    # Measure usage of CPU and memory and time at the start of ingestion
    start_cpu, start_memory = measure_CPU_and_memory_usage()
    # Start time
    start_time = time.time()
    args = parse_arguments()
    main(args.start_time, args.end_time, args.output_path)
    # End time
    end_time = time.time()
    end_cpu, end_memory = measure_CPU_and_memory_usage()

    # Calculate the used CPU and memory
    cpu_usage = end_cpu - start_cpu
    memory_usage = end_memory - start_memory

    # Calculate the ingestion time
    ingestion_time = end_time - start_time
    print(f"Data ingestion time: {ingestion_time} seconds")
    print(f"CPU usage: {cpu_usage}%")
    print(f"Memory usage: {memory_usage} MB")
