import pickle
import re
import sqlite3
from typing import List, Any

import nltk
from nltk import word_tokenize, WordNetLemmatizer
import pandas as pd

nltk.download(['punkt', 'wordnet'], quiet=True)


CATEGORIES_PATH = 'data/disaster_categories.csv'
MESSAGES_PATH = 'data/disaster_messages.csv'
DATABASE_PATH = 'data/disaster_response.db'
DATABASE_TABLE = 'messages'
MODEL_PATH = 'models/classifier.pkl'

URL_REGEX = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


def save_dataframe(df: pd.DataFrame, database: str = DATABASE_PATH, table: str = DATABASE_TABLE) -> None:
    conn = sqlite3.connect(database)
    df.to_sql(table, conn, index=False, if_exists='replace')
    conn.close()


def load_dataframe(database: str = DATABASE_PATH, table: str = DATABASE_TABLE) -> pd.DataFrame:
    conn = sqlite3.connect(database)
    df = pd.read_sql_query(f'select * from {table}', conn)
    conn.close()
    return df


def save_pickle(obj: Any, path: str = MODEL_PATH) -> None:
    pickle.dump(obj, open(path, 'wb'))


def load_pickle(path: str = MODEL_PATH) -> Any:
    return pickle.load(open(path, 'rb'))


def tokenize(text: str) -> List[str]:
    for url in re.findall(URL_REGEX, text):
        text = text.replace(url, 'urlplaceholder')

    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token).casefold().strip() for token in tokens]
