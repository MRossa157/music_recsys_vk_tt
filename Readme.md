# Рекомендательная система релевантных треков для пользователей
Репозиторий содержит:
- user-to-item модель (NDCG@20 0.110 на 30/04/2024)
- пока всё...

[Набор данных](https://www.kaggle.com/competitions/kkbox-music-recommendation-challenge/data)


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

2. Обучить user-2-item модель:

```python3 train_u2i_als.py```

3. Запустить main.py и проверить выдачу рекоменадий:

```python3 main.py```


# В планах:
- [ ] Добавить оценку метрики NDCG для NCF
- [ ] Добавить NCFRecommender
- [ ] Сделать makefile
