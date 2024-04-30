import pandas as pd


def get_users_and_items():
    df_members = pd.read_csv(r"app\dataset\members.csv")
    df_songs = pd.read_csv(r"app\dataset\songs.csv")

    df_test = pd.read_csv(r"app\dataset\test.csv")
    df_train = pd.read_csv(r"app\dataset\train.csv")

    all_users, all_items = [], []

    all_users.extend(df_members['msno'].unique().tolist())
    all_users.extend(df_test['msno'].unique().tolist())
    all_users.extend(df_train['msno'].unique().tolist())

    all_items.extend(df_songs['song_id'].unique().tolist())
    all_items.extend(df_test['song_id'].unique().tolist())
    all_items.extend(df_train['song_id'].unique().tolist())

    del df_songs, df_members, df_test, df_train
    return all_users, all_items