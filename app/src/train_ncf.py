import logging
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

from app.src.utils import get_train_data_ncf


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
    def __init__(self, df_train: DataFrame, all_users: list, all_items: list) -> None:
        """
        df_train (Pandas Dataframe) - Dataframe that you want to train the model on
        all_users (List) - List of all user ids (msno)
        all_items (List) - List of all song ids (song_id)
        """
        self.ALL_USERS = all_users
        self.ALL_ITEMS = all_items

        logging.info('Предобрабатываем данные')
        self.df_train = self.__prepare_data(df_train)

    def fit(self, max_epochs:int = 5):
        logging.info('Запустили обучение')
        num_users = len(self.ALL_USERS)
        num_items = len(self.ALL_ITEMS)

        model = NCF(num_users, num_items, self.df_train)
        trainer = pl.Trainer(max_epochs=max_epochs, logger=False)
        trainer.fit(model)
        logging.info('Сохраняем веса модели (чекпоинт) в папку app/src/weights')
        trainer.save_checkpoint(f"app/src/weights/NCF_result_epochs={max_epochs}.ckpt")

    def __get_maps(self, data: List | Tuple) -> dict:
        data_ids = dict(list(enumerate(data)))
        data_map = {u: uidx for uidx, u in data_ids.items()}
        return data_map

    def __prepare_data(self, df: DataFrame) -> DataFrame:
        """
        Remakes from the 'msno', 'song_id', 'target' dataframe a dataframe with columns ['user_id', 'item_id', 'target']
        """
        user_map = self.__get_maps(self.ALL_USERS)
        item_map = self.__get_maps(self.ALL_ITEMS)

        df['user_id'] = df['msno'].map(user_map)
        df['item_id'] = df['song_id'].map(item_map)

        df.dropna(subset=['item_id'], inplace=True)
        df['item_id'] = df['item_id'].astype(int)
        df = df[['user_id', 'item_id', 'target']]
        return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    df_train, ALL_USERS, ALL_ITEMS = get_train_data_ncf()
    trainer = TrainerNCF(df_train, all_users=ALL_USERS, all_items=ALL_ITEMS)
    trainer.fit(max_epochs=1)