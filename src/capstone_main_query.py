# NOTE: This code will *only* run internally on Mayo Clinic systems - it is *not locally operable*

# These lines presage any interaction with Mayo's proprietary cluster-based Google systems
# gcloud auth login
# gcloud auth application-default login

from google.cloud import bigquery  # Import Python-BigQuery interface API
from google.api_core import retry  # Ability to resend requests
import logging
import sys
from datetime import datetime

# Set up error logging
# We persist with "%-type" formatting to preserve backward compatibility
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('main_document_query.log'), logging.StreamHandler(sys.stdout)]
)

# Supply query credentials - these are 'spoofed' to avoid revealing actual Mayo Clinic projects/naming paradigms
project_id = 'qsr-user-7-radoncol-7cdp'  # Project ID code
dataspace_id = 'mayo_main_clinical'  # Dataspace to query
query_table_id = 'raw_clinical_documents_LIVE'  # Table to query - contains daily-updated clinical documents
output_table_id = 'radoncol_parser_data_temp'  # Temporary table where selected records will be stored

# First core process - Client initialization
try:
    bq_obj = bigquery.Client(project=project_id)  # Initialize BigQuery client
    logging.info('BigQuery client initialized successfully.')  # Log successful client initialization

except Exception as exc:
    # Catches authentication failures, not-found errors, and network connectivity problems during initialization
    logging.error(f'Failed to initialize BigQuery client. Error: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

# Set up query configuration
query_config = bigquery.QueryJobConfig(
    destination=output_table_id,
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE  # Delete any extant data before writing results
)

# Second core process - Define query content
query_string = f"""
SELECT PT_CLIN_NUM_RO_RESET AS pt_id, 
MAIN_CDOC_CLIN_DOC_UNQ_ID AS doc_id,
MAIN_CDOC_CLIN_DOC_DATE_CREATED AS doc_date,
MAIN_CDOC_CLIN_DOC_DATE_REVISED AS revision_date,
MAIN_CDOC_CLIN_DOC_CON_FULL AS text,
FROM {query_table_id}
WHERE 
    (LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '%radiation%'
    OR LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '%irradiation%'
    OR LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '%re-irradiation%'
    OR LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '%radiotherapy%'
    OR LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '%brachy%'
    OR LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '%radiosurgery%'
    OR LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '%chemoradiation%'
    OR LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '% rt %'
    OR LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE 'rt %'
    OR LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '% rt'
    OR LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) = 'rt') 
    AND DATE(MAIN_CDOC_CLIN_DOC_DATE_CREATED) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
"""

# Third core process - Extensive error handling for query attempt
try:
    # Execute query with retry decorator for transient errors like rate limits or network timeouts
    # The idea is that Google's tools can discriminate between internal "hiccups" and actual problems with the code
    @retry.Retry(predicate=retry.if_transient_error)
    def run_query():
        query_job = bq_obj.query(query_string, job_config=query_config)
        return query_job.result(timeout=180)  # Three minute timeout as required by Mayo IT

    results = run_query()  # Store query results

    # Fetch row count of query results
    destination_table = bq_obj.get_table(output_table_id)
    row_count = destination_table.num_rows

    # Log query completion and row count
    logging.info(f'Query Successful: {row_count} records loaded to temporary table {output_table_id}')

    # Top-line result validation
    if row_count == 0:  # Zero records returned indicate an issue with data ingestion or query timing
        logging.warning('WARNING: Query found no relevant documents for the previous day.')

except bigquery.exceptions.BadRequest as exc:
    # Catch SQL syntax errors, invalid project/dataset references, or incorrect query parameters
    logging.error(f'Query Syntax Error: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except bigquery.exceptions.Forbidden as exc:
    # Catch permission-related errors
    logging.error(f'Permission Denied: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except bigquery.exceptions.NotFound as exc:
    # Catch errors if referenced tables, datasets, or projects are not found
    logging.error(f'QuerySpace Not Found: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except TimeoutError:
    # Catch query execution timeouts
    logging.error('Query execution timed out after 3 minutes. Consult IT.')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except Exception as e:
    # Finally, catch all other errors
    logging.error(f'Unexpected error: {str(e)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script
