# NOTE: This code will *only* run internally on Mayo Clinic systems - it is *not locally operable*

# These lines presage any interaction with Mayo's proprietary cluster-based Google systems
# gcloud auth login
# gcloud auth application-default login

# Import Python-BigQuery interface API
from google.cloud import bigquery

# Supply query credentials - these are 'spoofed' to avoid revealing actual Mayo Clinic projects/naming paradigms
project_id = 'qsr-user-7-radoncol-7cdp'  # Project ID code
dataspace_id = 'mayo_main_clinical'  # Dataspace to query
query_table_id = 'raw_clinical_documents_LIVE'  # Table to query - contains daily-updated clinical documents
output_table_id = 'radoncol_parser_data_temp'  # Table where selected records will be stored
bq_obj = bigquery.Client(project=project_id)  # Initialize BigQuery client

# Set up query configuration
query_config = bigquery.QueryJobConfig(destination=output_table_id)

# Define query
query_string = f"""
SELECT PT_CLIN_NUM_RO_RESET AS pt_id, 
MAIN_CDOC_CLIN_DOC_UNQ_ID AS doc_id,
MAIN_CDOC_CLIN_DOC_DATE_ADDED AS record_date,
MAIN_CDOC_CLIN_DOC_CON_FULL AS text,
    CASE
        WHEN LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '%radiation%' THEN 'radiation'
        WHEN LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '%irradiation%' THEN 'irradiation'
        WHEN LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '%re-irradiation%' THEN 're-irradiation'
        WHEN LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '%radiotherapy%' THEN 'radiotherapy'
        WHEN LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '%brachy%' THEN 'brachy'
        WHEN LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '%radiosurgery%' THEN 'radiosurgery'
        WHEN LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '%chemoradiation%' THEN 'chemoradiation'
        WHEN LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '% rt %' 
          OR LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE 'rt %'
          OR LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) LIKE '% rt'
          OR LOWER(MAIN_CDOC_CLIN_DOC_CON_FULL) = 'rt' THEN 'rt'
        ELSE 'NO MATCH'
    END AS matched_key
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
    AND DATE(MAIN_CDOC_CLIN_DOC_DATE_ADDED) = DATE_SUB(CURRENT_DATE(), INTERVAL 1 DAY)
"""

# Start query and pass specified configuration
query_result = bq_obj.query(query_string).to_dataframe()

# Make the API request
query_job = bq_obj.query(query_string, job_config=query_config)

# Wait for the query to complete
query_job.result()

# Print completion/destination message
print(f'Query successful - results loaded to table {output_table_id}.')
