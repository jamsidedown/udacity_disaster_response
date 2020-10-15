import pickle
import re
import sqlite3
from typing import List, Tuple, Union

import nltk
from nltk import word_tokenize, WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


nltk.download(['punkt', 'wordnet'])


DATABASE_PATH = 'data/disaster_response.db'
MODEL_PATH = 'models/classifier.pkl'
TRAIN_TEST_DATA_PATH = 'models/training_testing_data.pkl'

URL_REGEX = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def load_data() -> Tuple[pd.Series, pd.Series, List[str]]:
    conn = sqlite3.connect(DATABASE_PATH)
    df = pd.read_sql_query("SELECT * from messages", conn)

    x = df['message']
    y = df[df.columns[4:]]

    return x, y


def tokenize(text: str) -> List[str]:
    for url in re.findall(URL_REGEX, text):
        text = text.replace(url, 'urlplaceholder')

    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token).casefold().strip() for token in tokens]


def build_model() -> Union[Pipeline, GridSearchCV]:
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('class', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
    }

    return GridSearchCV(pipeline, param_grid=parameters)


def display_results(model: GridSearchCV, y_test: pd.Series, y_pred: np.array) -> None:
    y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns)

    for column in y_test.columns.values:
        print(f'column {column}')
        print(classification_report(y_test[column], y_pred_df[column]))
    accuracy = (y_pred == y_test).mean().mean()

    print(f'{accuracy=}')
    print(f'{model.best_params_=}')


def evaluate_model(model: GridSearchCV, x_test: pd.Series, y_test: pd.Series) -> None:
    y_pred = model.predict(x_test)
    display_results(model, y_test, y_pred)


def save_model(model: GridSearchCV) -> None:
    pickle.dump(model, open(MODEL_PATH, 'wb'))


def load_model() -> GridSearchCV:
    return pickle.load(open(MODEL_PATH, 'rb'))


def main():
    print('==== Loading data ====')
    print(f'DATABASE: {DATABASE_PATH}')
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    print('==== Building model ====')
    model = build_model()

    print('==== Training model ====')
    model.fit(x_train, y_train)

    print('==== Evaluating model ====')
    evaluate_model(model, x_test, y_test)

    print('==== Saving model ====')
    print(f'MODEL: {MODEL_PATH}')
    save_model(model)

    print('==== Finished ====')


if __name__ == '__main__':
    main()