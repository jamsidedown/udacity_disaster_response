import re
import sqlite3

import nltk
from nltk import word_tokenize, WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


nltk.download(['punkt', 'wordnet'])


DATABASE_PATH = 'data/disaster_response.db'
MODEL_PATH = 'models/classifier.pkl'


def load_data():
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query("SELECT * from messages", conn)

    X = df['message']
    Y = df[df.columns[4:]]
    cat_names = Y.columns.values

    return X, Y, cat_names


def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    print('==== Loading data ====')
    print(f'DATABASE: {DATABASE_PATH}')
    X, Y, category_names = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    print('Data loaded')

    print(X.head())
    print(Y.head())
    print(category_names)

    print('==== Building model ====')
    model = build_model()
    print('Model built')

    print('==== Training model ====')
    model.fit(X_train, Y_train)
    print('Model trained')

    print('==== Evaluating model ====')
    evaluate_model(model, X_test, Y_test, category_names)
    print('Model evaluated')

    print('==== Saving model ====')
    print(f'MODEL: {MODEL_PATH}')
    save_model(model, MODEL_PATH)
    print('Model saved')


if __name__ == '__main__':
    main()