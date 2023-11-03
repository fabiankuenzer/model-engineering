import pandas as pd


def load_data():
    raw_data_1 = pd.read_csv('data/00 data-apr14.csv')
    raw_data_2 = pd.read_csv('data/01 data-may14.csv')
    raw_data_3 = pd.read_csv('data/02 data-jun14.csv')
    raw_data_4 = pd.read_csv('data/03 data-jul14.csv')
    raw_data_5 = pd.read_csv('data/04 data-aug14.csv')
    raw_data_6 = pd.read_csv('data/05 data-sep14.csv')

    raw_data = pd.concat([raw_data_1, raw_data_2, raw_data_3, raw_data_4, raw_data_5, raw_data_6])
    raw_data = raw_data.reset_index(drop=True)

    return raw_data
