import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression


# Загрузка данных
df = pd.read_csv("train_preprocessed/train_preprocessed_data.csv")

# Определение данных
X = df[["temp", "air_humidity", "pressure"]]
y = df["label"]

# Работа с моделью
model = LogisticRegression()
model.fit(X, y)

# Сохранение модели
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)