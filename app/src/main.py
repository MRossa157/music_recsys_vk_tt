import pandas as pd
from recommenders import ALSRecommender


def get_users_and_items(path_to_df_users, path_to_df_items):
    df_members = pd.read_csv(path_to_df_users)
    df_songs = pd.read_csv(path_to_df_items)

    all_users = df_members['msno'].unique().tolist()
    all_items = df_songs['song_id'].unique().tolist()

    del df_songs, df_members
    return all_users, all_items


# DEBUG PART
if __name__ == '__main__':
    # Получение списка всех пользователей и музыки
    path_to_df_users = r"app\dataset\members.csv"
    path_to_df_items = r"app\dataset\songs.csv"

    ALL_USERS, ALL_ITEMS = get_users_and_items(path_to_df_users, path_to_df_items)

    # Загрузка модели
    model_path = r"app/src/weights/als.npz"
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