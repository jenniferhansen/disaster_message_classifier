# run in cmd line:
# python train_classifier.py ../data/DisasterResponse.db classifier.pkl

# import packages & libraries
import sys
from functools import partial

import pandas as pd
import sqlite3
from sqlalchemy import create_engine

import pickle
import pandas as pd
import numpy as np

import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict
from sklearn.externals import joblib

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

#for ROC/AUC:
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

#from matplotlib import pyplot
#%matplotlib inline


def load_data(database_filepath):
    print('sqlite:///database_filepath')
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM CleanDataset", engine)
    X = df.message.values
    y = df.drop(['id','message','original','genre'], axis=1).values
    return X, y, df

def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # lemmatize andremove stop words
    clean_tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]

    return clean_tokens

def build_model():
    #build pipeline
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=partial(tokenize))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    model.set_params(
        vect__ngram_range = (1, 1),       #(1, 1)
        vect__max_df = 1.0,            #1.0
        vect__max_features = 10000,   #10000
        tfidf__use_idf = False,             #False
        clf__estimator__n_estimators = 100,  #n_estimators = number of trees in the foreset
        clf__estimator__min_samples_split = 2, #2 #min_samples_split = min number of data points placed in a node before the node is split
    )


    # specify parameters for grid search
    #parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': (True, False),
        #'clf__estimator__n_estimators': [10, 50, 100],  #n_estimators = number of trees in the foreset
        #'clf__estimator__min_samples_split': [2, 3, 4], #min_samples_split = min number of data points placed in a node before the node is split
        #'clf__estimator__max_features': [20,30,35],     #max number of features considered for splitting a node
        #'clf__estimator__max_depth': [5,20,50],         #max number of levels in each decision tree
        #'clf__estimator__min_samples_leaf': [],         #min number of data points allowed in a leaf node
        #'clf__estimator__bootstrap': [],                #method for sampling data points (with or without replacement)
        #'transformer_weights': (
        #    {'text_pipeline': 1},# 'starting_verb': 0.5},
        #    {'text_pipeline': 0.5},# 'starting_verb': 1},
        #    {'text_pipeline': 0.8}# 'starting_verb': 1},
        #)
    }

    # parameter discription:
    #n_estimators = number of trees in the foreset
    #min_samples_split = min number of data points placed in a node before the node is split
    #max number of features considered for splitting a node
    #max number of levels in each decision tree
    #min number of data points allowed in a leaf node
    #method for sampling data points (with or without replacement)
    #max_features = max number of features considered for splitting a node
    #max_depth = max number of levels in each decision tree
    #min_samples_leaf = min number of data points allowed in a leaf node
    #bootstrap = method for sampling data points (with or without replacement)


    # create grid search object
    #cv = GridSearchCV(model, param_grid=parameters)
    #return cv

    return model


def evaluate_model(model, X_test, Y_test, df):
    y_pred = model.predict(X_test)
    model_probs = model.predict_proba(X_test)
    return y_pred, model_probs


def save_model(model, model_filepath):
    # Export model as a pickle file

    # open the file for writing
    #fileObject = open(model_filepath,'wb')

    # this writes the object to the file named 'DisasterResponseModelLuke'
    #pickle.dump(model,fileObject)

    # here we close the fileObject
    #fileObject.close()

    #with open(model_filepath, 'wb') as file:
    #    pickle.dump(model, file)

    #filename = 'classifier.sav'
    filename = model_filepath
    joblib.dump(model, filename)





def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))

        ###X, Y, category_names = load_data(database_filepath)
        X, Y, df = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        ###evaluate_model(model, X_test, Y_test, category_names)
        evaluate_model(model, X_test, Y_test, df)

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
