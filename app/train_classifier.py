import os
import sys
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from .utils import DATABASE_PATH, MODEL_PATH, load_dataframe, tokenize, save_pickle


def load_data(database_path: str) -> Tuple[pd.Series, pd.Series, List[str]]:
    df = load_dataframe(database_path)

    x = df['message']
    y = df[df.columns[4:]]

    return x, y


def build_model() -> Pipeline:
    return Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, max_df=0.5)),
        ('tfidf', TfidfTransformer()),
        ('class', MultiOutputClassifier(KNeighborsClassifier()))
    ])


def build_model_gridsearch() -> GridSearchCV:
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

    return GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)


def display_results(model: Union[Pipeline, GridSearchCV], y_test: pd.Series, y_pred: np.array) -> None:
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


def main(database_path: str = DATABASE_PATH, model_path: str = MODEL_PATH) -> None:
    print('==== Loading data ====')
    print(f'DATABASE: {database_path}')
    x, y = load_data(database_path)

    print('==== Building model ====')
    model = build_model()

    print('==== Training model ====')
    model.fit(x, y)

    print('==== Saving model ====')
    print(f'MODEL: {model_path}')
    save_pickle(model, model_path)

    print('==== Finished ====')


def main_gridsearch(database_path: str = DATABASE_PATH, model_path: str = MODEL_PATH) -> None:
    print('==== Loading data ====')
    print(f'DATABASE: {database_path}')
    x, y = load_data(database_path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    print('==== Building model ====')
    model = build_model_gridsearch()

    print('==== Training model ====')
    model.fit(x_train, y_train)

    print('==== Evaluating model ====')
    evaluate_model(model, x_test, y_test)

    print('==== Saving model ====')
    print(f'MODEL: {model_path}')
    save_pickle(model, model_path)

    print('==== Finished ====')


if __name__ == '__main__':
    gridsearch = os.environ.get('GRIDSEARCH', 'false').casefold() == 'true'
    kwargs = {}

    if len(sys.argv) == 3:
        (database, model) = sys.argv[-2:]
        kwargs['database_path'] = database
        kwargs['model_path'] = model

    if gridsearch:
        main_gridsearch(**kwargs)
    else:
        main(**kwargs)
