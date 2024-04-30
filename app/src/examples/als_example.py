import pandas as pd

from app.src.recommenders import ALSRecommender
from app.src.utils import get_users_and_items

if __name__ == '__main__':
    # Получение списка всех пользователей и музыки
    ALL_USERS, ALL_ITEMS = get_users_and_items()

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