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

########################################################## NEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEW

def choose_trojan_locations(ht_count, ht_length, total_rows, total_columns,
                            averaging_lvl, ht_column_choice=None, initial_available_indices=None):
    """
    This function generates locations for the HTs to be placed.

    Arguments:  ht_count      -> Number of HT instances to be placed.
                ht_length     -> Number of rows to be allocated per HT instance. Same as T_ht.
                total_rows    -> Number of rows in the dataset where the HT will be placed.
                total_columns -> Number of columns in the dataset where the HT will be placed.
                averaging_lvl -> Number of elements used for calculating the column-wise moving average.

    Return:     * List of 'ht_count' number of tuples, where every tuple contains information
                  about the column and indices of a single HT instance.
                * Tuple with cached up information, which may be useful at a later stage.
                  Includes: (1) remaining_available_indices - Remaining indices where the HTs can be applied.
                            (2) ht_indices_all,             - Sorted indices where the HTs have been applied.
                            (3) ht_affected_indices_all.    - Sorted indices where the HTs have been applied
                                                              and where the HTs will affect the original value
                                                              in the dataset, due to the moving average effect.

    """

    buffer = averaging_lvl + ht_length + 111  # Number of indices blocked on either side of the first index of an HT.
    if initial_available_indices is None:
        initial_available_indices = np.arange(averaging_lvl, total_rows - ht_length)

    remaining_available_indices = initial_available_indices

    trojan_locations = []
    ht_indices_all = np.array([], dtype=np.int64)
    ht_affected_indices_all = np.array([], dtype=np.int64)
    for _ in range(ht_count):

        if ht_column_choice is None:
            ht_column = np.random.choice(range(total_columns))
        else:
            ht_column = ht_column_choice

        ht_index = np.random.choice(remaining_available_indices)    # Randomly choose an HT's first index.
        ht_indices = np.arange(ht_index, ht_index + ht_length)
        ht_indices_all = np.append(ht_indices_all, ht_indices)

        ht_affected_indices = np.arange(ht_index, ht_index + ht_length + averaging_lvl)
        ht_affected_indices_all = np.append(ht_affected_indices_all, ht_affected_indices)

        trojan_locations.append((ht_column, ht_indices))

        occupied_indices = np.arange(ht_index - buffer, ht_index + buffer + 1)
        remaining_available_indices = np.setdiff1d(remaining_available_indices, occupied_indices)

    ht_indices_all = np.sort(ht_indices_all)
    ht_affected_indices_all = np.sort(ht_affected_indices_all)

    cache = (remaining_available_indices, ht_indices_all, ht_affected_indices_all)

    return trojan_locations, cache


def generate_trojan_instance(ht_length, distribution_params, distribution_type='normal'):
    """
    This function generates a numpy array of power consumption values for a single HT instance.

    Arguments:  ht_length           -> Number of HT power values to be generated per HT instance. Same as T_ht.
                distribution_params -> Dictionary containing the parameters for the respective distribution.
                                       Keys: 'mean', 'sigma' for normal distribution,
                                             'min', 'max' for uniform distribution.
                distribution_type   -> String containing 'normal' or 'uniform'. The distribution type from
                                       which the HT's power consumption values will be drawn.

    Return:     * Array of length 'ht_length', containing HT's power consumption values.

    """

    if distribution_type is 'normal':
        mean = distribution_params["mean"]
        sigma = distribution_params["sigma"]
        ht_instance = np.random.normal(loc=mean, scale=sigma, size=(ht_length, 1))
    elif distribution_type is 'uniform':
        ht_min = distribution_params["min"]
        ht_max = distribution_params["max"]
        ht_instance = np.random.uniform(low=ht_min, high=ht_max, size=(ht_length, 1))
    else:
        raise ValueError("The distribution type should be 'normal' or 'uniform'.")

    return ht_instance


def matrix_of_trojans(total_rows, total_columns, trojan_locations, distribution_params, distribution_type='normal'):
    """
    This function generates a matrix of zeros with inserted power consumption
    values for all instances of HTs in their appropriate indices and columns.

    Arguments:  total_rows          -> Number of rows in the dataset where the HT will be placed.
                total_columns       -> Number of columns in the dataset where the HT will be placed.
                trojan_locations    -> List of tuples containing the column number and indices of the trojan.
                distribution_params -> Dictionary containing the parameters for the respective distribution.
                                       Keys: 'mean', 'sigma' for normal distribution,
                                             'min', 'max' for uniform distribution.
                distribution_type   -> String containing 'normal' or 'uniform'. The distribution type from
                                   which the HT's power consumption values will be drawn.

    Return:   * Sparse matrix of trojans' power consumption values. The shape matches that of the dataset
                in which the HTs are to be inserted.

    """

    ht_matrix = np.zeros((total_rows, total_columns))

    for col, rows in trojan_locations:
        ht_length = rows.size
        ht_instance = generate_trojan_instance(ht_length, distribution_params, distribution_type)
        ht_matrix[rows, col] = ht_instance

    return ht_matrix


def insert_all_trojans(dataset, averaging_lvl, ht_params_dictionary, initial_available_indices=None):
    """
    BLANK
    """
    total_rows, total_columns = dataset.shape
    ht_count  = ht_params_dictionary["ht_count"]
    ht_length = ht_params_dictionary["ht_length"]
    ht_column = ht_params_dictionary["ht_column_choice"]

    trojan_locations, cache = choose_trojan_locations(ht_count, ht_length, total_rows, total_columns,
                                                      averaging_lvl, ht_column_choice=ht_column,
                                                      initial_available_indices=initial_available_indices)
    _1, all_ht_indices, _3 = cache
    ht_distribution_params = ht_params_dictionary["ht_distribution_params"]
    ht_distribution_type   = ht_params_dictionary["ht_distribution_type"]

    ht_matrix = matrix_of_trojans(total_rows, total_columns, trojan_locations,
                                  ht_distribution_params, distribution_type=ht_distribution_type)

    infected_dataset = dataset + ht_matrix
    labels_y = np.ones((total_rows, 1), dtype=np.int32)
    labels_y[all_ht_indices] = -1

    infected_dataset = np.append(infected_dataset, labels_y, axis=1)

    return infected_dataset

# ---------- NEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEWNEW


def add_trojan_rows(data_set, i, num_of_trojan_rows, trojan_min, trojan_max, ht_column_choice=None):
    """ Take a random i from (0 : last - num_of_trojan_rows) and add HTs
        to rows (i : i + num_of_trojan_rows). Trojan power consumption is
        uniformly distributed in the range (trojan_min : trojan_max)"""

    trojan_power = np.random.uniform(low=trojan_min, high=trojan_max, size=(num_of_trojan_rows, 1))
    trojan_indexes = range(i, i + num_of_trojan_rows)
    data_set = np.r_[data_set]
    total_rows, total_columns = data_set.shape
    if ht_column_choice is None:
        available_columns = np.arange(total_columns)  # All columns
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

    # column_with_trojan = data_set[trojan_indexes, 0:1] + trojan_power
    # columns_without_trojan = data_set[trojan_indexes, 1:5]
    # trojan_rows = np.append(column_with_trojan, columns_without_trojan, axis=1)

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
