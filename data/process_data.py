import sqlite3

import pandas as pd
from pandas.io import sql


CATEGORIES_PATH = 'data/disaster_categories.csv'
MESSAGES_PATH = 'data/disaster_messages.csv'
DATABASE_PATH = 'data/disaster_response.db'


def load_data() -> pd.DataFrame:
    messages = pd.read_csv(MESSAGES_PATH)
    categories = pd.read_csv(CATEGORIES_PATH)

    return pd.merge(left=messages, right=categories, left_on='id', right_on='id')


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    categories_df = df['categories'].str.split(';', expand=True)
    category_headers = categories_df.iloc[0].apply(lambda x: x[:-2])
    categories_df.columns = category_headers

    for column in categories_df:
        categories_df[column] = categories_df[column].apply(lambda x: int(x[-1]))

    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories_df], sort=False, axis=1)

    return df.drop_duplicates()


def save_data(df: pd.DataFrame) -> None:
    conn = sqlite3.connect(DATABASE_PATH)
    df.to_sql('messages', conn, index=False, if_exists='replace')


def main():
    print('==== Loading data ====')
    print(f'MESSAGES: {MESSAGES_PATH}')
    print(f'CATEGORIES: {CATEGORIES_PATH}')
    df = load_data()
    print('Data loaded')

    print('==== Cleaning data ====')
    df = clean_data(df)
    print('Data cleaned')

    print('==== Saving database ====')
    print(f'DATABASE: {DATABASE_PATH}')
    save_data(df)
    print('Data saved')


if __name__ == '__main__':
    main()