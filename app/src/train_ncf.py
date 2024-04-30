import logging
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset


class MusicTrainDataset(Dataset):
    """MusicTrainDataset PyTorch Dataset for Training

    Args:
        ratings (pd.DataFrame): Dataframe which contains the 'user_id', 'item_id', 'target' columns

    """

    def __init__(self, ratings):
        self.users, self.items, self.labels = self.get_dataset(ratings)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

    def get_dataset(self, ratings):
        users, items, labels = [], [], []
        user_item_set = set(zip(ratings['user_id'], ratings['item_id']))

        for u, i in user_item_set:
            users.append(u)
            items.append(i)
            labels.append(1)

        return torch.tensor(users), torch.tensor(items), torch.tensor(labels)

class NCF(pl.LightningModule):
    """ Neural Collaborative Filtering (NCF)

        Args:
            num_users (int): Number of unique users
            num_items (int): Number of unique items
            ratings (pd.DataFrame): Dataframe containing the ratings for training
    """

    def __init__(self, num_users, num_items, ratings: DataFrame):
        super().__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=8)
        self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=8)
        self.fc1 = nn.Linear(in_features=16, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=1)
        self.ratings = ratings

    def forward(self, user_input, item_input):

        # Pass through embedding layers
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)

        # Concat the two embedding layers
        vector = torch.cat([user_embedded, item_embedded], dim=-1)

        # Pass through dense layer
        vector = nn.ReLU()(self.fc1(vector))
        vector = nn.ReLU()(self.fc2(vector))

        # Output layer
        pred = nn.Sigmoid()(self.output(vector))

        return pred

    def training_step(self, batch, batch_idx):
        user_input, item_input, labels = batch
        predicted_labels = self(user_input, item_input)
        loss = nn.BCELoss()(predicted_labels, labels.view(-1, 1).float())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def train_dataloader(self):
        return DataLoader(MusicTrainDataset(self.ratings),
                          batch_size=512, num_workers=0)
        # Если вы запускаете код на Google colab то можете выставить num_workers=5 (НЕ ПРОВЕРЕННО). В данный момент стоит 0, т.к:
        # jupyter notebook might not work properly with multiprocessing as documented (https://stackoverflow.com/a/71193241/16733101)


class TrainerNCF():
    def __init__(self, dataset_folder_path: str) -> None:
        logging.info('Загружаем данные о песнях и пользователях')
        self.df_songs = pd.read_csv(f'{dataset_folder_path}\\songs.csv')
        self.df_members = pd.read_csv(f'{dataset_folder_path}\\members.csv')

        self.ALL_USERS = self.df_members['msno'].unique().tolist()
        self.ALL_ITEMS = self.df_songs['song_id'].unique().tolist()

        del self.df_songs, self.df_members

        logging.info('Предобрабатываем данные')
        df_train = pd.read_csv(f'{dataset_folder_path}\\train.csv')
        user_map = self.get_maps(self.ALL_USERS)
        item_map = self.get_maps(self.ALL_ITEMS)
        df_train = self.__prepare_data(df_train, user_map, item_map)
        self.df_train = df_train[['user_id', 'item_id', 'target']]

    def get_maps(self, data: List | Tuple):
        data_ids = dict(list(enumerate(data)))
        data_map = {u: uidx for uidx, u in data_ids.items()}
        return data_map

    def __prepare_data(self, df: DataFrame, user_map: dict, item_map: dict):
        """
        Added to Pandas Dataframe new columns 'user_id' and 'item_id' based on exists columns 'msno' and 'song_id'
        """
        df['user_id'] = df['msno'].map(user_map)
        df['item_id'] = df['song_id'].map(item_map)

        df.dropna(subset=['item_id'], inplace=True)
        df['item_id'] = df['item_id'].astype(int)
        return df

    def fit(self, max_epochs:int = 5):
        logging.info('Запустили обучение')
        num_users = len(self.ALL_USERS)
        num_items = len(self.ALL_ITEMS)

        model = NCF(num_users, num_items, self.df_train)
        trainer = pl.Trainer(max_epochs=max_epochs, logger=False)
        trainer.fit(model)
        logging.info('Сохраняем веса модели (чекпоинт) в папку app/src/weights')
        trainer.save_checkpoint(f"app/src/weights/NCF_result_epochs={max_epochs}.ckpt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    trainer = TrainerNCF(r"app/dataset")
    trainer.fit(max_epochs=1)