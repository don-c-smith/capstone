"""
Pipeline Controller Script for Prior RT Record Classification Process
--------------------------------------------------
This script orchestrates the execution of the prior RT document classification pipeline, which processes clinical
documents to identify patients with prior radiation therapy.

Pipeline Steps:
1. Filter Records (capstone_filter_records.py):
   - Queries new clinical documents from the temporary BigQuery table
   - Filters out documents older than one week
   - Filters out documents already present in internal RadOnc database
   - Exports filtered records to .csv

2. Clean Data (capstone_clean_data.py):
   - Processes the filtered CSV file
   - Applies text preprocessing including:
     * Sentence tokenization
     * Keyword filtering
     * Clinical spell checking
     * Abbreviation expansion
     * Text normalization
   - Exports cleaned data to new .csv

3. Classify Records (capstone_classify_records.py):
   - Loads trained classifier and vectorizer from Joblib files
   - Processes cleaned text data
   - Makes predictions on new documents re: patient prior radiation
   - Exports results to final .csv for human review

Notable Features:
- Automatic retry logic for transient/unanticipated failures
- Comprehensive logging of all steps
- Pipeline execution monitoring
- Error handling with appropriate system exit codes

Usage note:
- This script should be placed in the same directory as the component scripts it orchestrates.
- It creates logs in the specified network directory.

Dependencies:
- Python 3.10+
- Access to Mayo Clinic network paths
- Permissions to use the internal MayoTools library
- All other dependencies required by component scripts as defined in the 'requirements.txt' file
"""
# Library imports
import subprocess
import logging
from datetime import datetime
import sys
import time
import os

# Set up logging system
log_dir = r'\\mayo_hpc\radoncol\prior_rt\pipeline_logs'
if not os.path.exists(log_dir):  # If the logging directory doesn't exist (which it should)
    os.makedirs(log_dir)  # Create the directory

log_file = os.path.join(log_dir, f'pipeline_execution_{datetime.now().strftime("%Y_%m_%d")}.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
                    )


def run_script(script_name: str, max_retries: int = 3, retry_delay: int = 120) -> bool:
    """
    This function executes a specified Python script using retry logic and logging.
    It is future-proof insofar asd additional scripts may be added to the pipeline and run in sequence.
    Args:
        script_name (str): Name of the script to run
        max_retries (int): Maximum number of retry attempts (Default: 3)
        retry_delay (int): Delay in seconds between retries (Default: 120 seconds/2 minutes)
    Returns:
        bool: True if a given script executes successfully and False otherwise
    """
    # First, we construct an absolute path to the script we want to run
    # __file__ fetches the script path, dirname fetches its directory, and we join with the script name
    script_path = os.path.join(os.path.dirname(__file__), script_name)

    # We attempt to run the script up to 'max_retries' times (Default: 3)
    for attempt in range(max_retries):
        try:
            # Log the start of execution for a given attempt
            logging.info(f'Attempting execution of {script_name}')

            # We use subprocess.run to execute the script as a separate process
            result = subprocess.run(
                ['python', script_path],  # Execute command (Parallel to common terminal .py execution)
                capture_output=True,  # Capture stdout/stderr rather than send to console
                text=True,  # Return output as string
                check=True  # Raise CalledProcessError if script exits with any errors
            )

            if result.stdout:  # Log any script output sent to stdout for debugging
                logging.info(f'Script Output: {result.stdout}')

            logging.info(f'{script_name} executed successfully.')  # Log any successful runs of the script
            return True  # And return True per the Docstring

        except subprocess.CalledProcessError as err:  # If the script exits with any errors
            # Log detailed error information:
            logging.error(f'Error in {script_name} (Attempt {attempt + 1}/{max_retries})')  # Log attempt number
            logging.error(f'Error code: {err.returncode}')  # Log error code
            logging.error(f'Error output: {err.stderr}')  # Log error output

            if attempt < max_retries - 1:  # If we haven't hit the 'max retries' value yet
                logging.info(f'Retrying script execution in {retry_delay} seconds...')  # Log the retry notification
                time.sleep(retry_delay)  # Pause and try to execute the script again

            else:  # If we exhaust all retry attempts
                # Log the failure notification
                logging.error(f'Maximum attempts reached for {script_name}. Pipeline halted. Please notify DSAA team.')
                return False  # And return False per the Docstring

        except Exception as exc:  # Catch any other errors - a safety net for unanticipated issues
            logging.error(f'An unexpected error occurred while running {script_name}: {str(exc)}')  # Log the error
            return False  # And return False per the Docstring


def main():
    """
    Main function to orchestrate the overall execution of the classifier pipeline.
    We run each script in the defined sequence.
    """
    # Define an ordered list of scripts to execute
    # NOTE: Order is important, as each script's integrity depends on the output of the previous script
    pipe_scripts = [
        'capstone_filter_records.py',  # Step 1: Filter records in temporary table, retrieve records, send to .csv
        'capstone_clean_data.py',  # Step 2: Clean and preprocess the data
        'capstone_classify_records.py'  # Step 3: Classify the processed records
    ]

    pipe_start = datetime.now()  # Record the exact start time for pipeline timing
    logging.info('Executing Pipeline...')  # Log the startup message

    for script in pipe_scripts:  # For each script in the pipeline, attempt execution and check return value
        # If any script fails (after maximum retries is reached), halt the pipeline
        if not run_script(script):
            logging.error('ERROR: Pipeline execution failed - halting process')  # Log the failure
            sys.exit(1)  # Exit with error status

        time.sleep(10)  # Pause between scripts to help prevent 'race' conditions re: file access

    # Calculate and log total pipeline execution time
    pipe_end = datetime.now()
    pipe_time = (pipe_start - pipe_end).total_seconds() / 60.0
    logging.info(f'Pipeline executed successfully. Runtime: {pipe_time:.2f} minutes.')


if __name__ == "__main__":
    try:
        main()  # Attempt to execute the main pipeline orchestration function

    except Exception as exc:  # Catch any unexpected errors in the main pipeline controller
        logging.error(f'Unexpected critical pipeline error: {str(exc)}')  # Log the error
        sys.exit(1)  # Exit with error status
