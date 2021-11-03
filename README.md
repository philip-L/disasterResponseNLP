# Disaster Response Pipeline Project

Udacity datascience project 2: A webapp to classify messages using a machine learning model trained on multi-labelled data, with natural language processing techniques.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Important Files

-./
    - ./app
        - run.py: Flask server to run the webapp
        - templates/: folder for html to render the webapp
    - ./data
        - process_data.py: ETL (extraction-transform-load) steps to prepare data 
        - disaster_messages.csv: data from Figure8 with message id and message text
        - disaster_categories.csv: prepared data from Figure8 with message id and categories indicated
    - ./models/train_classifier.py: Create, train and test the ML (machine-learning) model used for classifying messages
    - ETL Pipeline Preparation.ipynb: jupyter notebook for preparing data
    - ML Pipeline Preparation.ipynb: jupyter notebook for creating natural language processing (NLP) model

