import pandas as pd

from app.src.recommenders import ALSRecommender
from app.src.utils import get_users_and_items

if __name__ == '__main__':
    # Получение списка всех пользователей и музыки
    df_train = pd.read_csv(r'dataset/train.csv')
    df_test = pd.read_csv(r'dataset/test.csv')
    df_songs = pd.read_csv(r'dataset/songs.csv')
    df_members = pd.read_csv(r'dataset/members.csv')

    ALL_USERS, ALL_ITEMS = get_users_and_items([df_train, df_test, df_songs, df_members])

    # Загрузка модели
    model_path = r"src/weights/als.npz"
    model = ALSRecommender(model_path, user_ids=ALL_USERS, item_ids=ALL_ITEMS)

    # Выбор произвольного пользователя из трейн датасета
    df_rand_user = df_train.sample(1)

    # Получение рекомендаций
    user_msno = df_rand_user['msno'].item()
    user_id = model.msno_to_userid(user_msno)
    user_items_csr = model.create_user_item_csr_matrix(df_train, user_id = user_id)
    recs = model.get_recommendation(user_id, user_items_csr)

    print(recs)