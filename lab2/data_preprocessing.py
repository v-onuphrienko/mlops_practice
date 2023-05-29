import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# Чтение необработанных данных
df_train = pd.read_csv("train/train_data.csv")
df_test = pd.read_csv("test/test_data.csv")

# Обработка вероятных пропусков
df_train.fillna(df_train.mean(), inplace=True)
df_test.fillna(df_test.mean(), inplace=True)

# Работа с категориальными переменными
df_train = pd.get_dummies(df_train)
df_test = pd.get_dummies(df_test)

# Выбор функции для предобработки
corr_matrix = df_train.corr()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                  k = 1).astype(bool))

to_drop = [c for c in upper.columns if any(upper[c] > 0.9)]
df_train.drop(to_drop, axis=1, inplace=True)
df_test.drop(to_drop, axis=1, inplace=True)


scaler = StandardScaler()
df_train[["temp", "air_humidity", "pressure"]] = scaler.fit_transform(
    df_train[["temp", "air_humidity", "pressure"]])
df_test[["temp", "air_humidity", "pressure"]] = scaler.transform(
    df_test[["temp", "air_humidity", "pressure"]])

# Проверка наличия нужных папок
if not os.path.exists("train_preprocessed"):
    os.makedirs("train_preprocessed")
if not os.path.exists("test_preprocessed"):
    os.makedirs("test_preprocessed")

# Сохранение предобработанных данных
df_train.to_csv("train_preprocessed/train_preprocessed_data.csv", index=False)
df_test.to_csv("test_preprocessed/test_preprocessed_data.csv", index=False)