# Рекомендательная система релевантных треков для пользователей
Репозиторий содержит:
- user-to-item ALS модель (NDCG@20 = 0.110 на 30/04/2024)
- Neural Collaborative Filtering модель (NDCG@20 = 0.471 на 30/04/2024)
- вспомогательный переиспользуемый код

[Набор данных](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data)

Код написан так, что вполне успешно обучает модели на CPU

# Шаги по подготовке:
0. Скопировать данные в app/dataset
```
cd {REPO_ROOT}
mkdir -p app/dataset
cp /path/to/packed/data/*.zip ./app/dataset
```
1. Разархивировать архив(ы)

```python3 app/src/exctract_zip.py```

2. Обучить user-2-item ALS или NCF модель:

```python3 app/src/train_u2i_als.py```

```python3 app/src/train_ncf.py```

3. Запустить один из примеров кода и проверить выдачу рекоменадий:

```python3 app/src/examples/als_example.py```

```python3 app/src/examples/ncf_example.py```


# В планах:
- [X] Добавить оценку метрики NDCG@20 для NCF
- [X] Добавить NCFRecommender
- [ ] Сделать makefile
