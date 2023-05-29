import pandas as pd
import pickle


# Загрузка данных
df = pd.read_csv("test_preprocessed/test_preprocessed_data.csv")

# Загрузка модели
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Определение данных
X = df[["temp", "air_humidity", "pressure"]]
y = df["label"]

# Тестирование модели
accuracy = model.score(X, y)

# Вывод результатов
print(f"Результат работы модели = {accuracy}")