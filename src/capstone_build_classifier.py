import numpy as np
import pandas as pd
import re
import time
import joblib
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath: str) -> pd.DataFrame:
    """
    This function loads a .csv file into a Pandas dataframe, displays basic information about the loaded data,
    and returns the dataframe.
    Args:
        filepath (str): Path to the local .csv datafile.
    Returns:
        df (pd.DataFrame): Pandas dataframe containing loaded .csv data.
    """
    # Basic error handling
    try:
        df = pd.read_csv(filepath)
    except ImportError:
        print('Error importing data file. Please check file integrity and filetype and try again.')
    except FileNotFoundError:
        print(f'Error: File not found at {filepath}. Please check the file path and try again.')
    except pd.errors.EmptyDataError:
        print(f'Error: The file at {filepath} is empty. Please check the file content and try again.')
    except Exception as e:
        print(f'Error: An unexpected problem occurred - {str(e)}')

    # Print basic information about imported file
    print('Datafile Characteristics:')
    time.sleep(1)
    print(f'The datafile contains {len(df)} instances/records.')
    time.sleep(1)

    print('\nFeature names and datatypes:')
    time.sleep(1)
    for column, dtype in df.dtypes.items():
        print(f'{column} : {dtype}')
    time.sleep(1)

    print('\nCounts of Null values and Null rates by column:')
    time.sleep(1)
    for column in df.columns:
        null_count = df[column].isnull().sum()
        null_rate = (null_count / len(df)) * 100
        print(f'{column}: {null_count} ({null_rate:.2f}%)')

    # Return dataframe
    return df


def parse_text(text: str, keywords: list) -> str:
    """
    This function is applied to a single instance's clinical data text. It parses the text by removing any sentences
    which don't have at least one keyword in them (dimensionality reduction) and then further processes/prepares the
    text using the NLTK and RegEx libraries.
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

    # Lowercase all words
    text = text.lower()

    # Remove special characters (keeping hyphens and apostrophes)
    text = re.sub(r'[^a-zA-Z0-9\s\-\']', '', text)

    # Insert placeholder/stand-in for all numbers
    text = re.sub(r'\d+', '<num>', text)

    return text


def tokenize_and_vectorize_text(parsed_text: list) -> tuple:
    """
    This function tokenizes the parsed text and creates TF-IDF vectorized output for each instance.
    Args:
        parsed_text (list): The parsed text strings, passed as an iterable from a column in a Pandas dataframe
    Returns:
        A tuple of the actual vectorized output and the fitted TF-IDF vectorizer object
    """
    # Set up modified set of stopwords - domain-specific decisions
    my_stopwords = set(stopwords.words('english') + ['patient']) - {'no', 'not', 'without', 'never', 'denies'}

    def my_tokenizer(text: str):
        """This inner helper function tokenizes the parsed text at the word level and removes stopwords."""
        # Generate tokens
        tokens = word_tokenize(text, language='english')

        # Use list comprehension to return list of tokens with stopwords removed
        return [token for token in tokens if token not in my_stopwords]

    # Instantiate and define the TF-IDF vectorizer
    clin_text_vectorizer = TfidfVectorizer(tokenizer=my_tokenizer,  # Use helper function with my own stopwords
                                           lowercase=False,  # I already lowercased everything in the parser
                                           ngram_range=(1, 5))  # ngram range up to five words, adjust as needed

    # Fit vectorizer to the parsed text and generate sparse output
    vectorized_output = clin_text_vectorizer.fit_transform(parsed_text)

    # Return the vectorized output and the fitted vectorizer
    return vectorized_output, clin_text_vectorizer


def make_datasets(vectors, labels, test_size=0.3):
    """
    This function generates the training and test sets using the vectorized data and the values in the 'label' column.
    Args:
        vectors: The vectorized text from each instance.
        labels: The ground-truth labels for each instance.
        test_size: Proportion of dataset to be used in the test set.
    Returns:
        feature_train, feature_test, target_train, target_test: Training and test sets for use in modeling.
    """
    # Generate training and test sets
    feature_train, feature_test, target_train, target_test = train_test_split(vectors,
                                                                              labels,
                                                                              test_size=test_size,
                                                                              random_state=4)

    # Return training and test sets
    return feature_train, feature_test, target_train, target_test


def perform_classification(feature_train, feature_test, target_train, target_test, cv=5):
    """
    This function fits a logistic regressor to the vectorized training and test datasets and computes the accuracy
    metric and a classification report for the estimator.
    Args:
        feature_train, feature_test, target_train, target_test: Training and test sets for use in modeling.
        cv (int): Number of folds to use in cross-validation of the regressor.
    Returns:
        tuned_model: Best-performing logistic regressor as found by cross-validation.
        accuracy: Accuracy score on the test set.
        report: Classification report as a string.
        conf_mtrx: Confusion matrix as a NumPy array
    """
    # Define a parameter grid for GridSearch
    parameter_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['newton-cg', 'lbfgs', 'liblinear'],
        'class_weight': [None, 'balanced']
    }

    # Instantiate the regressor
    log_reg = LogisticRegression(random_state=4)

    # Perform grid search cross-validation
    grid_search = GridSearchCV(log_reg, parameter_grid, cv=cv, verbose=1)
    grid_search.fit(feature_train, target_train)

    # Fetch the best-performing model
    tuned_model = grid_search.best_estimator_

    # Make predictions on the test set
    test_preds = tuned_model.predict(feature_test)

    # Compute estimator performance metrics/reports
    accuracy = accuracy_score(target_test, test_preds)
    class_report = classification_report(target_test, test_preds)
    conf_mtrx = confusion_matrix(target_test, test_preds)

    # Print results of model evaluation
    print(f'Optimal Model Parameters: {grid_search.best_params_}')
    print(f'\nBest Cross-Validation Score: {grid_search.best_score_:.2f}')
    print(f'\nModel Accuracy on Test Set: {accuracy:.2f}')
    print('\nClassification Report:')
    print(class_report)

    # Display confusion matrix using Seaborn
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_mtrx, annot=True, fmt='d', cmap='coolwarm', annot_kws={'size': 18, 'weight': 'bold'})
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

    # Return tuned estimator and performance metrics/reports
    return tuned_model, accuracy, class_report, conf_mtrx


def assess_feature_importance(vectorizer, model, n_top_features=10):
    """
    Assess and visualize the feature importance values for the tuned logistic regressor.
    Args:
        vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer
        model (LogisticRegression): The trained logistic regression model
        n_top_features (int): Number of top features to display
    """
    def compute_importance_vals(vectorizer, model, n_top_features):
        """
        Compute feature importance values for the model trained on my TF-IDF vectors.
        """
        feature_names = vectorizer.get_feature_names_out()
        coeffs = model.coef_[0]

        feat_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(coeffs)
        })

        return feat_imp_df.sort_values('importance', ascending=False).head(n_top_features)

    top_features = compute_importance_vals(vectorizer, model, n_top_features)

    # Display most important features
    print(f'\n{n_top_features} Most Important Features:')
    print(top_features)

    # Visualize feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(top_features['feature'], top_features['importance'])
    plt.title(f'{n_top_features} Most Important Features')
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()  # Invert y-axis to show most important at the top
    plt.tight_layout()
    plt.show()


def main(filepath='sim_clin_data.csv'):
    save_path = r'\\mayo_hpc\radoncol\prior_rt\models'

    # Define keyword list
    keywords = ['radiation',
                'irradiation',
                're-irradiation',
                'radiotherapy',
                'brachy',
                'brachytherapy',
                'radiosurgery',
                'chemoradiation',
                'cancer',
                'tumor',
                'remission',
                'metastatic']

    # Load datafile
    df = load_data(filepath)

    # Parse text
    print('Beginning text parsing...')
    df['parsed_text'] = df['text'].apply(lambda instance: parse_text(instance, keywords))
    print('Text parsing complete.')

    # Tokenize and vectorize text
    print('Beginning tokenization and vectorization...')
    vectorized_output, clin_text_vectorizer = tokenize_and_vectorize_text(df['parsed_text'])
    print('Tokenization and vectorization complete.')

    # Create training and test sets
    print('Building training and test sets...')
    feature_train, feature_test, target_train, target_test = make_datasets(vectorized_output, df['label'])
    print('Training and test sets created.')

    # Perform classification
    print('Building and running the classifier...')
    tuned_model, accuracy, class_report, conf_mtrx = perform_classification(feature_train,
                                                                            feature_test,
                                                                            target_train,
                                                                            target_test)

    # Assess feature importance values
    print('Assessing feature importance values...')
    assess_feature_importance(clin_text_vectorizer, tuned_model)

    print('Saving trained model and vectorizer to network...')
    model_path = os.path.join(save_path, 'trained_regressor.joblib')
    vectorizer_path = os.path.join(save_path, 'trained_vectorizer.joblib')
    joblib.dump(tuned_model, model_path)
    joblib.dump(clin_text_vectorizer, vectorizer_path)
    print(f'Model saved to: {model_path}')
    print(f'Vectorizer saved to: {vectorizer_path}')

    # Return appropriate objects
    return df, vectorized_output, clin_text_vectorizer, tuned_model, accuracy, class_report, conf_mtrx


if __name__ == '__main__':
    (df,
     vectorized_output,
     clin_text_vectorizer,
     tuned_model,
     accuracy,
     class_report,
     conf_mtrx) = main()
