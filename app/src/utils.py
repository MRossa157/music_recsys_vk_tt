import os
import zipfile

import py7zr
from sklearn.metrics import ndcg_score
from tqdm import tqdm  # Для отображения прогресса


def _zip_extract(path, extract_to):
    with zipfile.ZipFile(path, 'r') as zfile:
        zfile.extractall(path=extract_to)
        for filename in tqdm(zfile.namelist(), desc="Разархивируем .zip"):
            full_path = os.path.join(extract_to, filename)
            recursive_extract(full_path, extract_to)

def _7z_extract(path, extract_to):
    with py7zr.SevenZipFile(path, mode='r') as zfile:
        zfile.extractall(path=extract_to)
        for filename in tqdm(zfile.getnames(), desc="Разархивируем .7z"):
            full_path = os.path.join(extract_to, filename)
            recursive_extract(full_path, extract_to)

def recursive_extract(zip_path, extract_to):
    if zipfile.is_zipfile(zip_path):
        _zip_extract(zip_path, extract_to)
    elif zip_path.endswith('.7z'):
        _7z_extract(zip_path, extract_to)

def extract_all(archive_path):
    # Определяем папку, в которой находится основной архив
    directory = os.path.dirname(archive_path)

    # Начинаем рекурсивное извлечение с основного архива
    recursive_extract(archive_path, directory)
    print("Все архивы успешно извлечены.")


def ndcg_20_score(y_true, y_score):
    return ndcg_score(y_true, y_score, k=20)