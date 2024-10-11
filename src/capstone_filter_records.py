# NOTE: This code will *only* run internally on Mayo Clinic systems - it is *not* locally operable

# These lines presage any interaction with Mayo's proprietary cluster-based Google systems
# gcloud auth login
# gcloud auth application-default login

# Import Python-BigQuery interface API
from google.cloud import bigquery
from datetime import datetime
import os

# Define global-scope information - this should almost never change
project_id = 'qsr-user-7-radoncol-7cdp'
dataspace_id = 'mayo_main_clinical'
temp_table_id = 'radoncol_parser_data_temp'
internal_table_id = 'radoncol_prior_rt_db'
network_path = r'\\mayo_hpc\radoncol\prior_rt\records\uncleared\unprocessed\dailies'

# Initialize BigQuery client
bq_obj = bigquery.Client(project=project_id)

# Define date-filtering query
date_filter_query = f"""
SELECT pt_id, 
       doc_id,
       record_date,
       text,
       matched_key
FROM `{project_id}.{dataspace_id}.{temp_table_id}`
WHERE DATE(record_date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 DAY)
"""

# TODO: Implement query error handling

# Execute query, load results into a dataframe
df_temp = bq_obj.query(date_filter_query).to_dataframe()

# TODO: Implement error handling for query and checking results against stable internal table
# Check document ID values against the values in the stable internal dataframe
doc_id_filter_query = f"""
SELECT DISTINCT doc_id
FROM `{project_id}.{dataspace_id}.{internal_table_id}`
"""

# Load distinct extant document IDs into a dataframe
doc_ids = bq_obj.query(doc_id_filter_query).to_dataframe()

# Filter out any documents which already exist in the stable internal database
df_filtered = df_temp[~df_temp['doc_id'].isin(doc_ids['doc_id'])]

# Print results of filters in terms of new records available for classification
print(f'{len(df_filtered)} new records passed the filters and are ready for preprocessing.')

# TODO: Implement error handling for .csv export
# Set up csv export
curr_date = datetime.now().strftime('%Y_%m_%d')
filename = f'radoncol_new_docs_{curr_date}.csv'
abs_path = os.path.join(network_path, filename)  # Define absolute path for file

# Export dataframe to csv
df_filtered.to_csv(abs_path, index=False)

print(f'Exported .csv file to {abs_path}')
