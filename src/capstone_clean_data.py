# Import required libraries
import os
import glob
from datetime import datetime
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
import logging
import sys
from MayoTools import clin_spell, expand_abv

# Set up error logging
# We persist with "%-type" formatting to preserve backward compatibility
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('data_cleaning_log.log'), logging.StreamHandler(sys.stdout)]
)

# Define keyword list
keywords = ['radiation',
            'irradiation',
            're-irradiation',
            'radiotherapy',
            'brachy',
            'brachytherapy',
            'radiosurgery',
            'chemoradiation',
            'rt']

# Define filepaths
input_path = r'\\mayo_hpc\radoncol\prior_rt\records\uncleared\unprocessed\dailies'
output_path = r'\\mayo_hpc\radoncol\prior_rt\records\uncleared\processed\dailies'

# Error handling for filepaths and file fetching
try:
    # Verify both filepaths exist
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'Input path not accessible at {input_path}')
    if not os.path.exists(output_path):
        raise FileNotFoundError(f'Output path not accessible at {output_path}')

    # Fetch all .csv files in the input folder
    csv_files = glob.glob(os.path.join(input_path, 'radoncol_new_docs_*.csv'))

    if not csv_files:  # If no files are present
        raise FileNotFoundError('No matching .csv-type files were found in the input directory.')

    # Sort the files based on extracted dates in descending order
    sorted_files = sorted(csv_files, key=lambda x:
    datetime.strptime(x.split('_')[-1].split('.')[0], '%Y_%m_%d'), reverse=True)

    # Fetch the newest file
    newest_file = sorted_files[0]

except FileNotFoundError as exc:
    logging.error(f'File system error: {str(exc)}')  # Catch incorrect paths or missing files
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except ValueError as exc:
    logging.error(f'Error parsing file dates: {str(exc)}')  # Catch datetime parsing errors
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except Exception as exc:
    logging.error(f'An unexpected error occurred during file handling: {str(exc)}')  # Catch all other errors
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

# Error handling for file reads
try:
    df = pd.read_csv(newest_file)  # Read the newest file
    logging.info(f'Loaded {len(df)} records from {os.path.basename(newest_file)}')

    if len(df) == 0:  # Length of zero means there's an error with the file or a pipeline failure
        logging.warning('WARNING: The input file contains no records. Check upstream processes.')

# Note that these exception clauses are Pandas-specific
except pd.errors.EmptyDataError:
    logging.error('Input .csv file is empty')  # Catch empty files
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except pd.errors.ParserError as exc:
    logging.error(f'Error parsing .csv file: {str(exc)}')  # Catch parsing errors
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except Exception as exc:
    logging.error(f'An unexpected error occurred while loading the .csv file: {str(exc)}')  # Catch all other errors
    sys.exit(1)  # Terminate program and signal failure to the coordinating script


def parse_text(text: str, keywords: list) -> str:
    """
    This function is applied to a single instance's clinical data text. It parses the text by removing any sentences
    which don't have at least one keyword in them (dimensionality reduction) and then further processes/prepares the
    text using the NLTK and RegEx libraries. It also applies the proprietary clinical spell-checker and
    abbreviation-expanding tools internally available in the MayoTools library.
    Args:
        text (str): The clinical note text to be parsed.
        keywords (list): The list of prior RT-relevant keywords.
    Returns
        text (str): The parsed text.
    """
    try:
        # Tokenize at sentence level with NLTK
        raw_sentences = sent_tokenize(text, language='english')

        # Keep only sentences containing keywords
        key_sentences = [sentence for sentence in raw_sentences if any(key in sentence.lower() for key in keywords)]

        # Rejoin key sentences into single string
        text = ' '.join(key_sentences)

        # Apply clin_spell to apply medical-focused spell-checking
        text = clin_spell(text)

        # Apply expand_abv to expand common medical abbreviations
        text = expand_abv(text)

        # Lowercase all words
        text = text.lower()

        # Remove special characters (keeping hyphens and apostrophes)
        text = re.sub(r'[^a-zA-Z0-9\s\-\']', '', text)

        # Insert placeholder/stand-in for all numbers
        text = re.sub(r'\d+', '<num>', text)

        return text

    except Exception as exc:
        # Log error but don't halt processing - return original text
        logging.error(f'WARNING: Error in NLP processing of text: {str(exc)}')
        print('ERROR PARSING TEXT. Returning original unprocessed string for review.')
        return text


try:
    # Apply the text preprocessing to the 'text' column for each document
    logging.info('Beginning text preprocessing...')
    df['text'] = df['text'].apply(lambda document: parse_text(document, keywords))
    logging.info('Text preprocessing complete.')

    # Create output filename
    input_filename = os.path.basename(newest_file)
    output_filename = input_filename.replace('.csv', '_PROCESSED.csv')
    output_filepath = os.path.join(output_path, output_filename)

    # Save the processed dataframe
    df.to_csv(output_filepath, index=False)
    logging.info(f'Processed data saved to: {output_filepath}')

except PermissionError as exc:
    # Catch permissions issues when writing file
    logging.error(f'Write permission denied when attempting to write output file: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except OSError as exc:
    logging.error(f'Error writing output file: {str(exc)}')  # Catch general file system errors
    sys.exit(1)  # Terminate program and signal failure to the coordinating script

except Exception as exc:
    logging.error(f'An unexpected error during data processing or export: {str(exc)}')
    sys.exit(1)  # Terminate program and signal failure to the coordinating script
