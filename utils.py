import pandas as pd
import numpy as np

def load_data(path):
    """
    - delete column with sum (in this particular case we don't need to measure the sum)
    - delete non-unique rows
    """
    # data = pd.read_csv(r'C:\Users\Gor\Desktop\ALL.txt', header=None, sep=';')
    data = pd.read_csv(path, header=None, sep=',')
    # data.columns = ['sum', '1', '2', '3', '4', '5', 'V_all', 'V1', 'V2', 'V3', 'V4', 'V5']
    # data = data.drop(['sum', 'V_all', 'V1', 'V2', 'V3', 'V4', 'V5'], axis=1)
    # data.columns = ['V_all', 'sum', '1', '2', '3', '4', '5']
    data.columns = ['1', '2', '3', '4', '5']
    # data = data.drop(['sum', 'V_all', 'V1', 'V2', 'V3', 'V4', 'V5'], axis=1)
    # data = data[:10000]
    # data = data.drop_duplicates()

    return data[['1', '2', '3', '4', '5']]


# def moving_average_numpy(data_set, periods=3):
#     """Return numpy array of the moving averages"""
#
#     weights = np.ones(periods) / periods
#     mov_av = np.convolve(data_set, weights, mode='valid')
#
#     return np.around(mov_av, decimals=2)

def add_trojan_rows(data_set, i, num_of_trojan_rows, trojan_min, trojan_max, ht_column_choice=None):
    """ Take a random i from (0 : last - num_of_trojan_rows) and add HTs
        to rows (i : i + num_of_trojan_rows). Trojan power consumption is
        uniformly distributed in the range (trojan_min : trojan_max)"""

    trojan_power = np.random.uniform(low=trojan_min, high=trojan_max, size=(num_of_trojan_rows, 1))
    trojan_indexes = range(i, i + num_of_trojan_rows)
    data_set = np.r_[data_set]
    total_rows, total_columns = data_set.shape
    if ht_column_choice is None:
        available_columns = np.arange(total_columns)  #All columns
        ht_column = np.random.choice(available_columns)  # Choose a random column to add the HT
    else:
        ht_column = ht_column_choice
    print("HT column: ", ht_column)
    # ht_free_columns = np.setdiff1d(available_columns, ht_column)
    column_with_trojan = data_set[trojan_indexes, ht_column:ht_column+1] + trojan_power
    columns_without_trojan_1 = data_set[trojan_indexes, 0:ht_column]
    columns_without_trojan_2 = data_set[trojan_indexes, ht_column+1:total_columns]
    trojan_rows = np.append(columns_without_trojan_1, column_with_trojan, axis=1)
    trojan_rows = np.append(trojan_rows, columns_without_trojan_2, axis=1)
    """column_with_trojan = data_set[trojan_indexes, 0:1] + trojan_power
    columns_without_trojan = data_set[trojan_indexes, 1:5]
    trojan_rows = np.append(column_with_trojan, columns_without_trojan, axis=1)"""
    infected_data_part_1 = np.append(data_set[range(i), 0:5], trojan_rows, axis=0)
    infected_data_all = np.append(infected_data_part_1, data_set[range(i + num_of_trojan_rows, total_rows), 0:5], axis=0)
    infected_data_all = pd.DataFrame(infected_data_all, columns=['1', '2', '3', '4', '5'])

    return infected_data_all


def moving_average_panda(data_set, periods=4, drop_initial_data=True):
    """Return panda data frame of the moving averages"""

    data_set['MA_Col1'] = data_set.iloc[:, 0].rolling(window=periods).mean()
    data_set['MA_Col2'] = data_set.iloc[:, 1].rolling(window=periods).mean()
    data_set['MA_Col3'] = data_set.iloc[:, 2].rolling(window=periods).mean()
    data_set['MA_Col4'] = data_set.iloc[:, 3].rolling(window=periods).mean()
    data_set['MA_Col5'] = data_set.iloc[:, 4].rolling(window=periods).mean()
    if drop_initial_data:
        data_set.drop(['1', '2', '3', '4', '5'], axis=1, inplace=True)
    data_set.drop(range(periods), inplace=True)

    return data_set


def split_to_train_test(split_ratio, input_data):
    """ Split the input data into train data and
     test data, with the split ratio given as input"""

    data = input_data.drop_duplicates()
    data = data.sample(frac = 1)
    data = np.r_[data]
    rows, columns = data.shape
    a = int(rows*split_ratio)
    train_data = data[0: a]
    test_data = data[a: rows+1]

    return train_data, test_data
