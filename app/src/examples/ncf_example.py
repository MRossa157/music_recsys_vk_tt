import pandas as pd

from app.src.recommenders import NCFRecommender
from app.src.utils import get_train_data_ncf

if __name__ == '__main__':
    # Получение списка всех пользователей и музыки
    df_train, ALL_USERS, ALL_ITEMS = get_train_data_ncf()

    model_path = r"app/src/weights/NCF_result_epochs=1.ckpt"
    model = NCFRecommender(model_path, user_ids=ALL_USERS, item_ids=ALL_ITEMS, df_train=df_train)

    # Выбор произвольного пользователя из трейн датасета
    df_rand_user = df_train.sample(1)

    # Получение рекомендаций
    user_msno = df_rand_user['msno'].item()
    user_id = model.msno_to_userid(user_msno)
    song_ids = model.get_user_song_ids(user_msno, df_train)
    items = [model.songid_to_item_id(i) for i in song_ids]

    # В силу того что в df_train и df_test представленны немного разные наборы песен, то маппинг песен по df_train от части НЕ АКТУАЛЕН для df_test
    # поэтому некоторые значения в списке items могут быть None. Т.к. модель не принимает такие значения, мы их не будем записывать в items
    items = [item_id for item_id in items if item_id is not None]

    recs = model.get_recommendation(user_id, items)

    print(recs)