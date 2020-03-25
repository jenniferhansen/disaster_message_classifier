# Disaster Response Pipeline Project

### Description:
# Final Pipeline to classifiy Disaster Response Messages

This projects consists of an ETL Pipeline and an ML Pipeline.
- The **ETL** part uses two CSV files and outputs a cleaned df and saves to an SQL database.
- The **ML pipeline** loads from the SQL database, initializes independent (X) and dependent variables (y),
does a train-test-split and then uses a multioutput randomforest classifier together with a TF-IDF transformer to classify 'disaster response messages' in 36 categories, e.g. "aid_related", "search_and_rescue", "food" or "water.

# Data from:
https://www.figure-eight.com/dataset/combined-disaster-response-data/


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or
   Go to http://127.0.0.1:3001/
