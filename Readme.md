# Рекомендательная система релевантных треков для пользователей
Репозиторий содержит:
- user-to-item ALS модель (NDCG@20 = 0.110 на 30/04/2024)
- Neural Collaborative Filtering модель (NDCG@20 = 0.471 на 30/04/2024)

[Набор данных](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data)

Код написан так, что вполне успешно обучает модели на CPU

# Шаги по подготовке:
0. Скопировать данные в app/dataset
```
cd {REPO_ROOT}
mkdir -p app/dataset
cp /path/to/packed/data/*.zip ./app/dataset
cd app/src
```
1. Разархивировать архив(ы)

```python3 exctract_zip.py```

2. Обучить user-2-item ALS модель:

```python3 train_u2i_als.py```

3. Запустить main.py и проверить выдачу ALS рекоменадий:

```python3 main.py```


# В планах:
- [X] Добавить оценку метрики NDCG@20 для NCF
- [X] Добавить NCFRecommender
- [ ] Сделать makefile
