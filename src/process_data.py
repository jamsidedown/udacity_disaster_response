import sys

import pandas as pd

from .utils import MESSAGES_PATH, CATEGORIES_PATH, DATABASE_PATH, save_dataframe


def load_data(messages_path: str, categories_path: str) -> pd.DataFrame:
    messages = pd.read_csv(messages_path)
    categories = pd.read_csv(categories_path)

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


def main(messages_path: str = MESSAGES_PATH, categories_path: str = CATEGORIES_PATH,
         database_path: str = DATABASE_PATH) -> None:
    print('==== Loading data ====')
    print(f'MESSAGES: {messages_path}')
    print(f'CATEGORIES: {categories_path}')
    df = load_data(messages_path, categories_path)

    print('==== Cleaning data ====')
    df = clean_data(df)

    print('==== Saving database ====')
    print(f'DATABASE: {database_path}')
    save_dataframe(df, database_path)

    print('==== Finished ====')


if __name__ == '__main__':
    if len(sys.argv) == 4:
        (messages, categories, database) = sys.argv[-3:]
        main(messages, categories, database)
    else:
        main()