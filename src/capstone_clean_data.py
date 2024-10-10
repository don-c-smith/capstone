# Import required libraries
import os
import glob
from datetime import datetime
import pandas as pd
import re
from nltk.tokenize import sent_tokenize
from MayoTools import clin_spell, expand_abv

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

# Fetch all csv files in the input folder
csv_files = glob.glob(os.path.join(input_path, 'radoncol_new_docs_*.csv'))

# Sort the files based on extracted dates in descending order
sorted_files = sorted(csv_files, key=lambda x:
                      datetime.strptime(x.split('_')[-1].split('.')[0], '%Y_%m_%d'), reverse=True)

# Fetch the newest file
newest_file = sorted_files[0]

# Read the newest file
df = pd.read_csv(newest_file)
print(f'Loaded {len(df)} records from {os.path.basename(newest_file)}')


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


# Apply the text preprocessing to the 'text' column for each document
df['text'] = df['text'].apply(lambda document: parse_text(document, keywords))

# Create the output filename
input_filename = os.path.basename(newest_file)
output_filename = input_filename.replace('.csv', '_PROCESSED.csv')
output_filepath = os.path.join(output_path, output_filename)

# Save the processed dataframe
df.to_csv(output_filepath, index=False)
print(f'Processed data saved to: {output_filepath}')
