import pandas as pd
from recommenders import ALSRecommender

# DEBUG PART
if __name__ == '__main__':
    # Получение списка всех пользователей и музыки
    df_songs = pd.read_csv(r"app\dataset\songs.csv")
    df_members = pd.read_csv(r"app\dataset\members.csv")

    ALL_USERS = df_members['msno'].unique().tolist()
    ALL_ITEMS = df_songs['song_id'].unique().tolist()

    del df_songs, df_members

    # Загрузка модели
    model_path = r"app\sandbox\weights\als.npz"
    model = ALSRecommender(model_path, user_ids=ALL_USERS, item_ids=ALL_ITEMS)

    # Выбор произвольного пользователя из тестового датасета
    df_test = pd.read_csv(r"app\dataset\test.csv")
    df_rand_user = df_test.sample(1)

    # Получение рекомендаций
    user_msno = df_rand_user['msno'].item()
    user_id = model.msno_to_userid(user_msno)
    user_items_csr = model.create_user_item_csr_matrix(df_test, user_id = user_id)
    recs = model.get_recommendation(user_id, user_items_csr)

    print(recs)