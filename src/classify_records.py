import os
import joblib
from datetime import datetime
from capstone_build_classifier import load_data, tokenize_and_vectorize_text

# Define paths
model_path = r'\\mayo_hpc\radoncol\prior_rt\models\trained_regressor.joblib'
vectorizer_path = r'\\mayo_hpc\radoncol\prior_rt\models\trained_vectorizer.joblib'
input_dir = r'\\mayo_hpc\radoncol\prior_rt\records\uncleared\processed\dailies'
output_dir = r'\\mayo_hpc\radoncol\prior_rt\records\to_review'

# TODO: Implement model and vectorizer load error handling
# Load model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# TODO: Implement file load error handling
# Fetch all csv files and sort them by date
csv_files = [file for file in os.listdir(input_dir) if file.endswith('_PROCESSED.csv')]
newest_file = sorted(csv_files, key=lambda x: datetime.strptime(x.split('_')[-2], '%Y_%m_%d'), reverse=True)[0]

# TODO: Implement tokenization error handling
# Process the most recent file
input_path = os.path.join(input_dir, newest_file)
df = load_data(input_path)
vectorized_output, _ = tokenize_and_vectorize_text(df['text'])
predictions = model.predict(vectorized_output)
df['predicted_label'] = predictions

# TODO: Implement export-and-save error handling
# Save results
output_filename = newest_file.replace('_PROCESSED.csv', '_CLASSIFIED.csv')
output_path = os.path.join(output_dir, output_filename)
df.to_csv(output_path, index=False)

print(f'Made classification predictions for {newest_file} and saved results.')
