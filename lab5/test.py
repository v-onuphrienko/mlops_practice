
import pytest
import os
import pandas as pd
import pickle


datasets = {}
files = os.listdir('./Data/')
for df_file in files:
    print(df_file)
    datasets[df_file] = pd.read_csv('./Data/' + df_file)

pkl_filename = 'model.pkl'
with open(pkl_filename, 'rb') as file_:
    model = pickle.load(file_)


def test_metric_crash():
    '''Тестируем метрику'''
    for dataset in datasets.keys():
        test_y = datasets[dataset]['Y']
        test_X = datasets[dataset].drop('Y', axis=1)
        assert model.score(test_X, test_y) > 0.98


def test_std():
    '''Тестируем дисперсию'''
    for dataset in datasets.keys():
        assert datasets[dataset]['B'].describe()['std'] < 50


def test_median():
    '''Ну и медиану, чтобы хоть один тест прошёл для примера'''
    for dataset in datasets.keys():
        assert 150 < datasets[dataset]['B'].describe()['50%'] < 300
