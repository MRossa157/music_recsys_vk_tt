import logging
import os

import implicit
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import coo_matrix

BEST_PARAMS = {'factors': 40, 'iterations': 3, 'regularization': 0.01}

def train(coo_train, factors=200, iterations=15, regularization=0.01, show_progress=True):
    model = implicit.als.AlternatingLeastSquares(factors=factors,
                                                 iterations=iterations,
                                                 regularization=regularization)
    model.fit(coo_train, show_progress=show_progress)
    return model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    dataset_path = r"app\dataset"
    # Считывание данных
    logging.info('Считываем данные')
    df_train = pd.read_csv(rf'{dataset_path}\train.csv')

    # мапинг msno и song_id на нормальные индексы
    logging.info("Добавляем колоки 'user_id' и 'item_id'")
    ALL_USERS = df_train['msno'].unique().tolist()
    ALL_ITEMS = df_train['song_id'].unique().tolist()

    user_ids = dict(list(enumerate(ALL_USERS)))
    item_ids = dict(list(enumerate(ALL_ITEMS)))

    user_map = {u: uidx for uidx, u in user_ids.items()}
    item_map = {i: iidx for iidx, i in item_ids.items()}

    df_train['user_id'] = df_train['msno'].map(user_map)
    df_train['item_id'] = df_train['song_id'].map(item_map)

    # Обучение
    logging.info('Подготавливаем данные для обучения')
    row = df_train['user_id'].values
    col = df_train['item_id'].values
    data = np.ones(df_train.shape[0])
    coo_train = coo_matrix((data, (row, col)), shape=(len(ALL_USERS), len(ALL_ITEMS)))
    csr_train = coo_train.tocsr()

    logging.info('Запускаем обучение')
    model = train(csr_train, **BEST_PARAMS)

    logging.info('Сохраняем веса в папку app/src/weights')
    if not os.path.exists(r'app/src/weights'):
        os.makedirs(r'app/src/weights')

    model.save(r'app/src/weights/als.npz')