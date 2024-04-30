import logging
from typing import Tuple

import pandas as pd
from pandas import DataFrame


def get_users_and_items(dfs: list[DataFrame]) -> Tuple[list]:
    """
    Method that takes as input DataFrames with columns msno, song_id and returns a list of all unique users(msno) and songs(song_id)

    Params:

    dfs (List (Pandas Dataframe)) - List of Pandas DataFrames
    """
    logging.info('Загружаем данные о песнях и пользователях')

    all_users, all_items = [], []
    for df in dfs:
        # В некоторых Датафреймах (например df_songs) нет информации о пользователях
        # поэтому мы просто ловим исключение и ничего с ним не делаем
        try:
            all_users.extend(df['msno'].unique().tolist())
        except KeyError:
            pass
        try:
            all_items.extend(df['song_id'].unique().tolist())
        except KeyError:
            pass

    # Удалим дубликаты
    all_users, all_items = list(set(all_users)), list(set(all_items))

    return all_users, all_items


def get_train_data_ncf():
    df_train = pd.read_csv(r"app\dataset\train.csv")
    all_users, all_items = get_users_and_items([df_train])
    return df_train, all_users, all_items
