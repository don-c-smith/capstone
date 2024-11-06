# NOTE: This code will *only* run internally on Mayo Clinic systems - it is *not* locally operable

# These lines presage any interaction with Mayo's proprietary cluster-based Google systems
# gcloud auth login
# gcloud auth application-default login

# Import Python-BigQuery interface API
from google.cloud import bigquery
from datetime import datetime
import logging
import sys
import os

# Set up error logging
# We persist with "%-type" formatting to preserve backward compatibility
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('record_filtering_log.log'), logging.StreamHandler(sys.stdout)]
)

# Define global-scope information - this should almost never change
project_id = 'qsr-user-7-radoncol-7cdp'
dataspace_id = 'mayo_main_clinical'
temp_table_id = 'radoncol_parser_data_temp'
internal_table_id = 'radoncol_prior_rt_db'
network_path = r'\\mayo_hpc\radoncol\prior_rt\records\uncleared\unprocessed\dailies'

try:
    bq_obj = bigquery.Client(project=project_id)  # Initialize BigQuery client
    logging.info('BigQuery client initialized successfully.')  # Log successful client initialization

except Exception as exc:
    # Catches authentication failures, not-found errors, and network connectivity problems during initialization
    logging.error(f'Failed to initialize BigQuery client. Error: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

# Define date-filtering query content
date_filter_query = f"""
SELECT pt_id, 
       doc_id,
       doc_date,
       revision_date,
       text,
FROM `{project_id}.{dataspace_id}.{temp_table_id}`
WHERE DATE(revision_date) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 WEEK)  # Exclude any revisions older than 1 week
"""

try:
    # Execute query, load results into a dataframe
    logging.info('Executing date filter query. Please wait.')
    df_temp = bq_obj.query(date_filter_query).to_dataframe()
    logging.info(f'Retrieved {len(df_temp)} records from temporary table.')

    if len(df_temp) == 0:
        # No records found in the temporary table indicates an upstream issue
        logging.warning('WARNING: No records found in temporary table. Check upstream processes and contact IT.')

except bigquery.exceptions.BadRequest as exc:
    # Catch SQL syntax errors or incorrect table references
    logging.error(f'Invalid syntax in document filter: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except bigquery.exceptions.NotFound as exc:
    # In case for some reason the temporary table doesn't exist
    logging.error(f'ERROR: Temporary table not found. Check if previous step completed successfully. {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except Exception as exc:
    # Catch all other unexpected errors
    logging.error(f'An unexpected error occurred during document filtering: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

# Error handling for internal database document ID cross-comparison
try:
    # Check document ID values against the values in the stable internal dataframe
    doc_id_filter_query = f'SELECT DISTINCT doc_id FROM `{project_id}.{dataspace_id}.{internal_table_id}`'

    # Load distinct extant document IDs into a dataframe
    logging.info('Retrieving existing document IDs...')
    doc_ids = bq_obj.query(doc_id_filter_query).to_dataframe()
    logging.info(f'There are {len(doc_ids)} existing internal document IDs')

except bigquery.exceptions.NotFound as exc:
    # In case the internal table can't be found
    logging.error(f'Internal database table not found: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except Exception as exc:
    logging.error(f'Error retrieving existing document IDs: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

try:
    # Filter out any documents which already exist in the stable internal database
    df_filtered = df_temp[~df_temp['doc_id'].isin(doc_ids['doc_id'])]

    # Print results of filters in terms of new records available for classification
    record_count = len(df_filtered)
    logging.info(f'{record_count} new records passed the filters and are ready for preprocessing.')

    if record_count == 0:
        # All records were filtered out - might indicate an issue
        logging.warning('ATTENTION: All new records were filtered out. There are no new documents to process.')

except Exception as exc:
    # Catch any other document ID comparison errors
    logging.error(f'An unexpected error occurred during document ID comparison: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

# Error handling for .csv export
try:
    # Set up csv export
    curr_date = datetime.now().strftime('%Y_%m_%d')
    filename = f'radoncol_new_docs_{curr_date}.csv'
    abs_path = os.path.join(network_path, filename)  # Define absolute path for file

    # Verify network path exists
    if not os.path.exists(network_path):
        raise FileNotFoundError(f'Network path {network_path} is not accessible.')

    # Export dataframe to csv
    df_filtered.to_csv(abs_path, index=False)
    logging.info(f'Successfully exported {record_count} filtered documents to {abs_path}')

except FileNotFoundError as exc:
    # Catch inaccessible network paths
    logging.error(f'Network path error: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except PermissionError as exc:
    # Catch permission issues when writing file
    logging.error(f'Permission denied when writing to network path: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except Exception as exc:
    # Catch any other file-writing errors
    logging.error(f'An unexpected error occurred during file export: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script
