import pandas as pd
import os
import joblib
from datetime import datetime
from capstone_build_classifier import load_data, tokenize_and_vectorize_text
import logging
import sys

# Set up error logging
# We persist with "%-type" formatting to preserve backward compatibility
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('record_classification_log.log'), logging.StreamHandler(sys.stdout)]
)

# Define paths
model_path = r'\\mayo_hpc\radoncol\prior_rt\models\trained_regressor.joblib'
vectorizer_path = r'\\mayo_hpc\radoncol\prior_rt\models\trained_vectorizer.joblib'
input_dir = r'\\mayo_hpc\radoncol\prior_rt\records\uncleared\processed\dailies'
output_dir = r'\\mayo_hpc\radoncol\prior_rt\records\to_review'

# Error handling for pathfinding and model loading
try:
    # Verify all paths exist
    for path in [model_path, vectorizer_path, input_dir, output_dir]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Required path not accessible at {path}')

    # Load classifier and vectorizer
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        logging.info('Successfully loaded classifier and vectorizer.')
    except Exception as exc:
        logging.error(f'Error while attempting to load classifier and vectorizer: {str(exc)}')
        raise  # Re-raise the same exception to be caught by the outer error handling block

except FileNotFoundError as exc:
    logging.error(f'Path error: {str(exc)}')  # Catch non-existent file
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except Exception as exc:
    logging.error(f'An unexpected error occurred during initialization: {str(exc)}')  # Catch all other errors
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

# Error handling for .csv fetching and sorting
try:
    # Fetch .csv files and sort them by date
    csv_files = [file for file in os.listdir(input_dir) if file.endswith('_PROCESSED.csv')]

    if not csv_files:  # If no files are available
        raise FileNotFoundError('No processed .csv files were found in the input directory.')

    # Fetch the newest file
    newest_file = sorted(csv_files,
                         key=lambda x: datetime.strptime(x.split('_')[-2], '%Y_%m_%d'),
                         reverse=True)[0]
    logging.info(f'Located newest file: {newest_file}')

except IndexError:
    logging.error('Error accessing sorted files - check file names')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except ValueError as exc:
    logging.error(f'Error parsing file dates: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except Exception as exc:
    logging.error(f'An unexpected error occurred during file handling: {str(exc)}')  # Catch all other errors
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

# Error handling for running the classifier
try:
    # Process the most recent file
    input_path = os.path.join(input_dir, newest_file)

    # Load and validate the datafile
    df = load_data(input_path)
    if len(df) == 0:  # If the file is empty
        raise ValueError('ERROR: The input file contains no records. Check upstream processes.')

    logging.info(f'Loaded {len(df)} records from {newest_file}.')  # Log successful data upload

    # Vectorize text and make predictions
    try:
        vectorized_output, _ = tokenize_and_vectorize_text(df['text'])
        predictions = model.predict(vectorized_output)
        df['predicted_label'] = predictions
        logging.info(f'Successfully made predictions for {len(predictions)} new records.')

    except Exception as exc:
        logging.error(f'Error during text vectorization or classification: {str(exc)}')  # Catch all other errors
        raise

except pd.errors.EmptyDataError:
    logging.error('Input CSV file is empty.')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except ValueError as exc:
    logging.error(f'Data validation error occurred: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except Exception as exc:
    logging.error(f'An unexpected error occurred during data processing: {str(exc)}')  # Catch all other errors
    sys.exit(1)

# Error handling for saving results
try:
    # Save results
    output_filename = newest_file.replace('_PROCESSED.csv', '_CLASSIFIED.csv')
    output_path = os.path.join(output_dir, output_filename)

    # Save the dataframe
    df.to_csv(output_path, index=False)
    logging.info(f'Successfully saved classification results to {output_path}')
    print(f'Made classification predictions for {newest_file} and saved results.')

except PermissionError as exc:
    logging.error(f'Write permission denied when attempting to write output file: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except OSError as exc:
    logging.error(f'OS error when writing output file: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except Exception as exc:
    logging.error(f'An unexpected error occurred during file export: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script
