from abc import ABC
from typing import List, Tuple

import torch
from implicit.cpu.als import AlternatingLeastSquares
from pandas import DataFrame
from scipy.sparse import coo_matrix

from app.src.train_ncf import NCF


class BaseRecommender(ABC):
    def __init__(self, model_path: str, user_ids: list, item_ids: list, ) -> None:
        self.user_ids = user_ids
        self.item_ids = item_ids

        user_ids_dict = dict(list(enumerate(self.user_ids)))
        item_ids_dict = dict(list(enumerate(self.item_ids)))

        self.user_map = {u: uidx for uidx, u in user_ids_dict.items()}
        self.item_map = {i: iidx for iidx, i in item_ids_dict.items()}

    def get_recommendation(self) -> str:
        raise NotImplementedError

    def msno_to_userid(self, msno:str):
        return self.user_map.get(msno, None)

    def songid_to_item_id(self, songid):
        return self.item_map.get(songid, None)

class ALSRecommender(BaseRecommender):
    def __init__(self, model_path: str, user_ids: list, item_ids: list) -> None:
        """
        model_path (str) = Path to .npz checkpoint file from lib 'implicit' ALS model
        """
        super().__init__(model_path, user_ids, item_ids)
        if not model_path.endswith(".npz"):
            raise ValueError("Путь к модели должен содержать файл с расширением .npz")
        self.model = AlternatingLeastSquares.load(model_path)

    def get_recommendation(self, userid, user_items, N=10, filter_already_liked_items=True, filter_items=None, recalculate_user=False, items=None) -> tuple:
        '''
        Parameters
        userid (Union[int, array_like]) – The userid or array of userids to calculate recommendations for

        user_items (csr_matrix) – A sparse matrix of shape (users, number_items). This lets us look up the liked items and their weights for the user. This is used to filter out items that have already been liked from the output, and to also potentially recalculate the user representation. Each row in this sparse matrix corresponds to a row in the userid parameter: that is the first row in this matrix contains the liked items for the first user in the userid array.

        N (int, optional) – The number of results to return

        filter_already_liked_items (bool, optional) – When true, don’t return items present in the training set that were rated by the specified user.

        filter_items (array_like, optional) – List of extra item ids to filter out from the output

        recalculate_user (bool, optional) – When true, don’t rely on stored user embeddings and instead recalculate from the passed in user_items. This option isn’t supported by all models.

        items (array_like, optional) – Array of extra item ids. When set this will only rank the items in this array instead of ranking every item the model was fit for. This parameter cannot be used with filter_items

        Returns
        Tuple of (itemids, scores) arrays. For a single user these array will be 1-dimensional with N items.
        '''
        return self.model.recommend(userid, user_items, N=N, filter_already_liked_items=filter_already_liked_items,
                                   filter_items=filter_items, recalculate_user=recalculate_user, items=items)

    def create_user_item_csr_matrix(self, df: DataFrame, user_id: int = None, msno: str = None):
        """
        Создает CSR матрицу для указанного пользователя на основе DataFrame с информацией о взаимодействиях пользователей и товаров.

        Параметры:
        df (pd.DataFrame): DataFrame с столбцами user_id и item_id.
        user_id (int): ID пользователя, для которого нужно создать матрицу.
        msno (str): msno (аналог user_id) пользователя

        Возвращает:
        scipy.sparse.csr_matrix: CSR матрица, где строка представляет пользователя, а столбцы - товары.
        """

        if not user_id and not msno:
            raise ValueError(f'Вы должны передать один из параметров: user_id = {user_id} или msno = {msno}')

        if msno:
            user_id = self.msno_to_userid(msno)

        df['user_id'] = df['msno'].map(self.user_map)
        df['item_id'] = df['song_id'].map(self.item_map)

        df_user = df[df['user_id'] == user_id]

         # Если у пользователя нет взаимодействий, возвращаем пустую CSR матрицу
        if df_user.empty:
            return coo_matrix((0, 0)).tocsr()

        df_user['item_id'] = df_user['item_id'].astype('category')

        row = [0] * len(df_user)  # Так как у нас один пользователь, все элементы будут в одной строке
        col = df_user['item_id'].cat.codes  # Индексы для столбцов
        data = [1] * len(df_user)

        num_items = df_user['item_id'].cat.categories.size

        matrix_coo = coo_matrix((data, (row, col)), shape=(1, num_items))

        return matrix_coo.tocsr()

    def similar_items(self, itemid) -> tuple:
        return self.model.similar_items(itemid)

    def similar_users(self, userid) -> tuple:
        return self.model.similar_users(userid)


class NCFRecommender(BaseRecommender):
    def __init__(self, model_path: str, user_ids: list, item_ids: list, df_train: DataFrame) -> None:
        """
        model_path (str) = Path to .ckpt checkpoint file from PyTorch Lightning

        user_ids (List[str|int]) = List of all users

        item_ids (List[str|int]) = List of all items

        df_train (Pandas Dataframe) = Train Dataframe
        """
        super().__init__(model_path, user_ids, item_ids)
        if not model_path.endswith(".ckpt"):
            raise ValueError("Путь к модели должен содержать файл с расширением .ckpt")

        ratings = self.__prepare_data(df_train)

        self.model = NCF.load_from_checkpoint(model_path, num_users=len(self.user_ids), num_items=len(self.item_ids), ratings=ratings)

    def get_recommendation(self, user_id: int, items: list) -> str:
        """
        user_id (int) = User ID (not 'msno' from DataFrame)

        items_targets List(int) = List of item_ids (not 'song_id' from DataFrame)

        Returns
        Tuple of (itemids, scores) arrays. For a single user these array will be 1-dimensional with N items.
        """
        # Прогнозы модели для данного пользователя
        items_tensor = torch.tensor(items, dtype=torch.long)
        user_tensor = torch.tensor([user_id] * len(items), dtype=torch.long)
        with torch.no_grad():
            predictions = self.model(user_tensor, items_tensor).flatten().numpy()

        return items_tensor.flatten().numpy(), predictions

    def get_user_song_ids(self, msno: str, df: DataFrame):
        items = df[df['msno'] == msno]['song_id'].tolist()
        return items

    def __get_maps(self, data: List | Tuple) -> dict:
        data_ids = dict(list(enumerate(data)))
        data_map = {u: uidx for uidx, u in data_ids.items()}
        return data_map

    def __prepare_data(self, df: DataFrame) -> DataFrame:
        """
        Remakes from the 'msno', 'song_id', 'target' dataframe a dataframe with columns ['user_id', 'item_id', 'target']
        """
        user_map = self.__get_maps(self.user_ids)
        item_map = self.__get_maps(self.item_ids)

        df['user_id'] = df['msno'].map(user_map)
        df['item_id'] = df['song_id'].map(item_map)

        df.dropna(subset=['item_id'], inplace=True)
        df['item_id'] = df['item_id'].astype(int)
        df = df[['user_id', 'item_id', 'target']]
        return df