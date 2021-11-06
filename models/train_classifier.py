import sys
# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer # different from nltk.stem.wordnet ?
from nltk.corpus import stopwords
import pickle
import re

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    """ load data from database, drop NaN's, split into X and y """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messageCategories',engine)
    df = df.dropna()
    df = df.related[df.related != 2]
    X = df['message']
    y = df.loc[:,'related':]
    category_names = y.columns
    
    return X, y, category_names

def tokenize(text):
    """Tokenization function used in CountVectorizer to count words"""
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)

    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():
    """Use Pipeline to build model"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {  'clf__estimator__n_estimators': [10, 20],
                'clf__estimator__min_samples_split': [2, 5]
             }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """test model and print precion, recall, f1-score and accuracy"""
    y_pred = model.predict(X_test)
    
    # convert to a DataFrame
    y_pred = pd.DataFrame(y_pred, columns=category_names)
    
    # iterate through the columns and calling sklearn's classification_report on each
    # report accuracy, precision and recall for each output category of the dataset
    for col in y_pred.columns:
        report = classification_report(Y_test[col], y_pred[col])
        precision, recall, f1 = report[report.find('weight'):].split()[2:5]
        print(col + ':\n' + f'precision: {precision}' + ' ' + f'recall: {recall}' + ' ' + f'f1-score: {f1}')
     
    Y_test = Y_test.reset_index(drop=True)  # indexes need to be identical in y_pred and Y_test
    accuracy = (y_pred == Y_test).mean()
    print("Accuracy:\n", accuracy)

def save_model(model, model_filepath):
    """save model as pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """main function"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()