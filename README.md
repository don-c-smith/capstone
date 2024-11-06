# Capstone Project - Simulated Mayo Clinic ML Pipeline
This repository contains the production scripts, sample data, and classifier construction .py file for the implementation of an end-to-end ML pipeline at Mayo Clinic.

**BACKGROUND:**
When patients seek out radiation therapy (RT) for the treatment of cancer, a crucial piece of medical information for the radiation oncology department is whether or not said patient has received prior radiation therapy. Considerations of cumulative radiation exposure, site-based side effects and professional practices, etc. all affect the ways in which Mayo prescribes their own courses of radiation therapy. Unfortunately, because many patients don't possess their complete medical history and it can be hard to obtain that information from other facilities (since Mayo sees patients from all over the world), the best resource Mayo often has to assess whether a patient has received prior radiation comes from the clinical notes taken by various medical professionals during patient interviews, which are stored as consistently-formatted text in Mayo's main patient queryspace.

As such, if a patient's past medical records are not available, a team of human reviewers at Mayo reads through the clinical notes looking for evidence of prior radiation. The team achieves approximately a 92% accuracy rate in reviewing the text (which is contextually-excellent in the medical field), but obviously this process is slow and costly. The goal of this project is to build a scalable supervised ML classifying estimator which, based on the content of a patient's clinical notes, can rapidly classify that patient as prior RT/no prior RT with a higher level of accuracy than is currently being achieved by the human-review team.

**DATA COLLECTION AND PREPARATION:**
Clinical note data may be pulled from Mayo's main queryspace, but it must be handled, sent, and stored in strict accordance with data governance policies. (See 'docs' folder for more information.) The data, once retrieved, are filtered based on date of document upload and the presence in the text of one or more radiotherapy-relevant keywords which have been defined by Mayo's oncologists. Once the records have been filtered, the sentences containing the RT-relevant keywords are extracted from the full text, and those 'key' sentences are processed using various methods involving the Regex and NLTK libraries, along with a few proprietary text-processing tools unique to (and only available at) Mayo Clinic. (See 'docs' folder for more information.) The processed key sentences are then vectorized using a TF-IDF method and sent to the classifier.  

**CLASSIFIER CONSTRUCTION:**
A key consideration, given the number of clinical documents created daily at Mayo, is that the text-fetching, text-processing, text-vectorizing, and text-classifying tools be as scalable and as computationally lightweight as possible. Mayo's HPC resources are extensive but highly-utilized - it is a "large but loaded" system whose resources must be managed carefully and are assigned by a specialist Research IT team. Therefore, building the "lightest" ML pipeline possible will not only make the process faster and more assessable, but will also be more easily "fit into" the daily schedule of available computational resources.

To that end, the text-processing and text-vectorizing code is written using libraries and methods which are known to scale well and to exhibit transparency. While a more sophisticated text-processing library (such as a member of the BERT family) might be more powerful than a more traditional TF-IDF vectorizer, it is much slower, and orders of magnitude more "expensive", to construct and train. Additionally, rather than initially reaching for a powerful but computationally-costly ensemble estimator, our first-principle classifying model (a logistic regressor) was chosen for similar speed-, cost-, and transparency-related reasons. We were prepared to build and operationalize more complex and expensive models if our lightweight system's performance was sub-par, but fortunately, our classifier performs very well using these less-costly systems.

Our classifier was trained using simulated LLM-generated records which exactly mimic the "top-line" summary medical review comments present in the queryspace. We used a large sample of records to define the parameters of the records which the LLM used as a template, and instructed the model to create misleading "trap" records intended to try to "fool" the classifier. The need for using simulated data to train the classifier is obvious, given that actual medical records may not be used in a system whose code and supporting data are publicly available. (Please see the 'docs' folder for more details on the simulated clinical data.)

**INITIAL OUTCOMES:**
Initial outcomes for classifier performance on simulated data are extremely encouraging. The classifier, despite being lightweight and using minimal cross-validation, performed a near-perfect classification of unseen test-set data, with accuracy of 99% and similar performance on other classifying metrics. (See the confusion matrix in the 'figures' folder.) Therefore, the "lightweight" model performs extremely well - better than the current performance of human reviewers - despite its computational simplicity and the relatively uncomplex vectorization process.

**NOTES ON PRODUCTION SCRIPTS:**
The production-level scripts in the 'src' folder represent the four core steps of the classifier pipeline:
1. Fetch Records (capstone_main-query.py)
   - Queries the main clinical record QuerySpace for new clinical documents containing RT-relevant keywords
   - Ensures that these are newly-added records, not revisions of old records
   - Sends the retrieved records to a temporary table in accordance with data governance policy

2. Filter Records (capstone_filter_records.py):
   - Queries new clinical documents from the temporary BigQuery table
   - Filters out documents older than one week
   - Filters out documents already present in internal RadOnc database
   - Exports filtered records to .csv

3. Clean Data (capstone_clean_data.py):
   - Processes the filtered CSV file
   - Applies text preprocessing including:
     * Sentence tokenization
     * Keyword filtering
     * Clinical spell checking
     * Abbreviation expansion
     * Text normalization
   - Exports cleaned data to new .csv

4. Classify Records (capstone_classify_records.py):
   - Loads trained classifier and vectorizer from Joblib files
   - Processes cleaned text data
   - Makes predictions on new documents re: patient prior radiation
   - Exports results to final .csv for human review

*NOTE:*
- The 'capstone_build_classifier' script shows the build-and-train process used to create the vectorizer and classifier objects used as the 'heart' of the ML pipeline. 
- This script also saves the created vectorizer and predictive model as Joblib files.
- The 'execute_classifier_pipeline' script runs the 'filter', 'clean', and 'classify' scripts in sequence.
- The main_query script is run every day at a time specified by Mayo IT. The other three production scripts are called by the 'execute_classifier_pipeline' control script.
- All scripts contain extensive error-handling and logging implementations to assist in transparency and debugging efforts.

**DEPENDENCIES AND TOOL USAGE:**
- Most scripts in the pipeline run using common data science tools and libraries.
- Required libraries are listed in the requirements.txt file.
- NOTE: The capstone_clean_data script relies on proprietary data analysis tools which are only available when connected to the internal Mayo Clinic network.
- For IP-related legal reasons, I can neither discuss in detail nor produce code for those tools on a public repository.
- The data-cleaning steps relevant to those proprietary tools can simply be 'commented out' and the scripts will remain operable.
- As such, by downloading the simulated data, training the classifier, saving the vectorizer and classifier, modifying the filepaths, and running the control script, anyone may simulate a single run of this pipeline by providing additional "unseen" simulated clinical data. 