import pandas as pd
import numpy as np
import sys


def load_data(path):
    """
    Load the CSV file data from the given path and store in a pandas data frame.

    Arguments:  path    -> Path to the CSV data file with HT-clean raw data points.

    Return:     * Pandas data frame.

    """

    data = pd.read_csv(path, header=None, sep=',')
    data.columns = ['1', '2', '3', '4', '5']

    # data = data.drop(['sum', 'V_all', 'V1', 'V2', 'V3', 'V4', 'V5'], axis=1)
    # data = data.drop_duplicates()
    # data = data[:10000]

    return data


def choose_trojan_locations(ht_count, ht_length, total_rows, total_columns,
                            averaging_lvl, ht_column_choice=None, initial_available_indices=None):
    """
    This function generates locations for the HTs to be placed.

    Arguments:  ht_count      -> Number of HT instances to be placed.
                ht_length     -> Number of rows to be allocated per HT instance. Same as T_ht.
                total_rows    -> Number of rows in the dataset where the HT will be placed.
                total_columns -> Number of columns in the dataset where the HT will be placed.
                averaging_lvl -> Number of elements used for calculating the column-wise moving average.
                ht_column_choice -> 'None' or integer smaller than the number of columns.
                initial_available_indices -> 'None' or numpy array of available/desired index range open to
                                              apply HTs. Elements must be within dataset row range.

    Return:     * List of 'ht_count' number of tuples, where every tuple contains information
                  about the column and indices of a single HT instance.
                * Tuple with cached up information, which may be useful at a later stage.
                  Includes: (1) remaining_available_indices - Remaining indices where the HTs can be applied.
                            (2) ht_indices_all,             - Sorted list of indices where the HTs have been applied.
                            (3) ht_affected_indices_all.    - Sorted list of indices where the HTs have been applied
                                                              and where the HTs will affect the original value in
                                                              the dataset, due to the moving average effect.

    """

    buffer = averaging_lvl + ht_length + 77  # Number of indices blocked on either side of the first index of an HT.
    if initial_available_indices is None:
        initial_available_indices = np.arange(averaging_lvl, total_rows - ht_length)

    remaining_available_indices = initial_available_indices

    trojan_locations = []
    ht_indices_all = np.array([], dtype=np.int64)
    ht_affected_indices_all = np.array([], dtype=np.int64)
    for _ in range(ht_count):

        if ht_column_choice is None:
            ht_column = np.random.choice(range(total_columns))
        elif ht_column_choice in range(total_columns):
            ht_column = ht_column_choice
        else:
            raise ValueError("The ht_column_choice should be 'None' or integer smaller than the number of columns.")
            # sys.exit("Provide correct ht_column_choice.")

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

    print("choose_trojan_locations: Done!")

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
        ht_instance = np.random.normal(loc=mean, scale=sigma, size=(ht_length, ))
    elif distribution_type is 'uniform':
        ht_min = distribution_params["min"]
        ht_max = distribution_params["max"]
        ht_instance = np.random.uniform(low=ht_min, high=ht_max, size=(ht_length, ))
    else:
        raise ValueError("The distribution type should be 'normal' or 'uniform'.")
        # sys.exit("Provide correct distribution type.")

    print(".", end="")

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

    Return:     * Sparse matrix of trojans' power consumption values. The shape matches that of the dataset
                  in which the HTs are to be inserted.

    """

    ht_matrix = np.zeros((total_rows, total_columns))

    for col, rows in trojan_locations:
        ht_length = rows.size
        ht_instance = generate_trojan_instance(ht_length, distribution_params, distribution_type)
        ht_matrix[rows, col] = ht_instance

    print("\nmatrix_of_trojans: Done!")

    return ht_matrix


# UPDATED VERSION
# def matrix_of_trojans(ht_matrix, total_rows, total_columns, trojan_locations, ht_distribution):
#     """
#     This function adds a single trojan instance to the ht_matrix, which is a sparse matrix with
#     inserted power consumption values for all instances of HTs in their appropriate indices and columns.
#
#     Arguments:  ht_matrix        -> Sparse matrix containing power consumption values for all existing instances of HTs.
#                 total_rows       -> Number of rows in the dataset where the HT will be placed.
#                 total_columns    -> Number of columns in the dataset where the HT will be placed.
#                 trojan_locations -> List of tuples containing the column number and indices of the trojan.
#                 ht_distribution  -> Tuple describing the ht power consumption distribution. Contains
#                                     either ('normal', mean_value, sigma) or ('uniform', min_value, max_value)
#
#     Return:     * Sparse matrix containing power consumption values for all existing instances of HTs.
#                   The shape matches that of the dataset in which the HTs are to be inserted.
#
#     """
#
#     ht_matrix = ht_matrix
#     distribution_type = ht_distribution[0]
#     if distribution_type is "normal":
#       distribution_params = ht_distribution = {"mean": ht_distribution[1], "sigma": ht_distribution[2]}
#     elif distribution_type is "uniform"
#       distribution_params = ht_distribution = {"min": ht_distribution[1], "max": ht_distribution[2]}
#
#     for col, rows in trojan_locations:
#         ht_length = rows.size
#         ht_instance = generate_trojan_instance(ht_length, distribution_params, distribution_type)
#         ht_matrix[rows, col] = ht_instance
#
#     print("\nmatrix_of_trojans: Done!")
#
#     return ht_matrix


def insert_all_trojans(dataset, averaging_lvl, ht_params_dictionary, initial_available_indices=None):
    """
    This function takes the HT-clean dataset and overlays the HT (sparse) matrix, which
    should have the same shape as the HT-clean dataset. This function also adds an extra
    column for the labels y which indicates the rows with added HTs.

    Arguments:  dataset             -> The HT-clean raw dataset.
                averaging_lvl       -> Number of elements used for calculating the column-wise moving average.
                ht_params_dictionary     -> Dictionary containing parameter keys:
                                                ht_count -  Number of HT instances to be placed.
                                                ht_length - Number of HT power values to be generated per HT
                                                            instance. Same as T_ht.
                                                ht_column_choice - 'None' or integer smaller than the number of columns.
                                                ht_distribution  -  Dictionary containing the parameters for
                                                                         the respective distribution. Keys:
                                                                         'mean', 'sigma' for normal distribution,
                                                                         'min', 'max' for uniform distribution.
                                                ht_distribution_type   - String containing 'normal' or 'uniform'.
                                                                         The distribution type from which the HT's
                                                                         power consumption values will be drawn.
                initial_available_indices     -> 'None' or numpy array of available/desired index range open to
                                                  apply HTs. Elements must be within dataset row range.

    Return:   * The HT-infected dataset with an extra column 'labels' indicating the rows without (1) and with (-1) HTs.
              * Tuple with cached up information, which may be useful at a later stage.
                Includes: (1) remaining_available_indices - Remaining indices where the HTs can be applied.
                          (2) ht_indices_all,             - Sorted list of indices where the HTs have been applied.
                          (3) ht_affected_indices_all.    - Sorted list of indices where the HTs have been applied
                                                            and where the HTs will affect the original value in
                                                            the dataset, due to the moving average effect.

    """
    total_rows, total_columns = dataset.shape
    ht_count  = ht_params_dictionary["ht_count"]
    ht_length = ht_params_dictionary["ht_length"]
    ht_column = ht_params_dictionary["ht_column_choice"]

    trojan_locations, cache = choose_trojan_locations(ht_count, ht_length, total_rows, total_columns,
                                                      averaging_lvl, ht_column_choice=ht_column,
                                                      initial_available_indices=initial_available_indices)
    _1, all_ht_indices, _3 = cache
    ht_distribution = ht_params_dictionary["ht_distribution"]
    ht_distribution_type   = ht_params_dictionary["ht_distribution_type"]

    ht_matrix = matrix_of_trojans(total_rows, total_columns, trojan_locations,
                                  ht_distribution, distribution_type=ht_distribution_type)

    # ht_matrix = np.zeros((total_rows, total_columns))
    # remaining_available_indices = initial_available_indices
    # ht_distribution_tuples = ht_params_dictionary["ht_distribution_tuples"]
    # ht_lengths = ht_params_dictionary["ht_lengths"]
    # all_ht_indices = []
    # all_ht_affected_indices = []
    # caches = []
    # for ht_length in ht_lengths:
    #     for ht_distribution_tuple in ht_distribution_tuples:
    #
    #         trojan_locations, cache = choose_trojan_locations(ht_count, ht_length, total_rows, total_columns,
    #                                                           averaging_lvl, ht_column_choice=ht_column,
    #                                                           initial_available_indices=remaining_available_indices)
    #         remaining_available_indices, new_ht_indices, new_ht_affected_indices = cache
    #         all_ht_indices.extend(new_ht_indices)
    #         all_ht_affected_indices.extend(new_ht_affected_indices)
    #         caches.extend(cache)
    #         ht_matrix = matrix_of_trojans(ht_matrix, total_rows, total_columns,
    #                                       trojan_locations, ht_distribution_tuple)

    infected_dataset = dataset + ht_matrix
    labels_y = np.ones((total_rows, 1), dtype=np.int32)
    labels_y[all_ht_indices] = -1

    infected_dataset = np.append(infected_dataset, labels_y, axis=1)

    print("insert_all_trojans: Done!")

    return infected_dataset, cache  # return infected_dataset, all_ht_affected_indices, caches


def moving_average_panda(dataset, avg_lvl=5, drop_initial_data=True):
    """
    This function calculates the moving averages of every column and 
    appends to the dataset in a new column in the respective row. It
    also provides the option of dropping the initial columns and
    leaving only the columns consisting of the moving averages.

    Arguments:  dataset -> The dataset.
                avg_lvl -> Integer specifying the size of the window for calculating the moving average.
                drop_initial_data -> 'True' for dropping and 'False' for leaving the initial columns. 

    Return:     * Pandas data frame.

    """
    dataset = pd.DataFrame(dataset, columns=['1', '2', '3', '4', '5', 'labels_y'])
    dataset['MA_Col1'] = dataset.iloc[:, 0].rolling(window=avg_lvl).mean()
    dataset['MA_Col2'] = dataset.iloc[:, 1].rolling(window=avg_lvl).mean()
    dataset['MA_Col3'] = dataset.iloc[:, 2].rolling(window=avg_lvl).mean()
    dataset['MA_Col4'] = dataset.iloc[:, 3].rolling(window=avg_lvl).mean()
    dataset['MA_Col5'] = dataset.iloc[:, 4].rolling(window=avg_lvl).mean()
    if drop_initial_data:
        dataset.drop(['1', '2', '3', '4', '5'], axis=1, inplace=True)
    dataset.drop(range(avg_lvl), inplace=True)

    print("moving_average_panda: Done!")
    dataset = dataset[['1', '2', '3', '4', '5', 'MA_Col1', 'MA_Col2', 'MA_Col3', 'MA_Col4', 'MA_Col5', 'labels_y']]

    return dataset


def train_dev_test_set(dataset, dev_test_ratio, ht_affected_indices_all):
    """
    This function divides a dataset into three datasets: train, development, test.
    Train dataset contains HT-clean points, while development and test datasets
    contain 50% HT-clean and 50% HT-infected points.
    The division is done based on the given dev_test_ratio tuple.

    Arguments:  dataset -> The dataset (keep the original indexing, until the ht_affected_indices_all are removed).
                dev_test_ratio -> Tuple floats containing development and test set size
                ratios to the combined size. Should add up to 1.
                ht_affected_indices_all -> Sorted list of indices where the HTs have been applied
                                           and where the HTs have affected the original value in
                                           the dataset, due to the moving average effect.

    Return:     * Training, development and testing datasets.

    """
    dev_ratio, test_ratio = dev_test_ratio

    all_ht_data = dataset.loc[dataset['labels_y'] == -1]
    all_ht_clean_data = dataset.drop(ht_affected_indices_all)

    all_ht_clean_data.drop_duplicates(inplace=True)

    all_ht_data = all_ht_data.sample(frac=1)
    all_ht_clean_data = all_ht_clean_data.sample(frac=1)

    all_ht_data.reset_index(inplace=True)
    all_ht_clean_data.reset_index(inplace=True)

    rows_ht, cols_ht = all_ht_data.shape

    trojans_dev  = all_ht_data.iloc[:int(dev_ratio*rows_ht), :]
    trojans_test = all_ht_data.iloc[int(dev_ratio*rows_ht):, :]

    clean_dev  = all_ht_clean_data.iloc[:int(dev_ratio*rows_ht), :]
    clean_test = all_ht_clean_data.iloc[int(dev_ratio*rows_ht):rows_ht, :]

    train_set = all_ht_clean_data.iloc[rows_ht:, :].drop(['index'], axis=1)
    dev_set = pd.concat([trojans_dev, clean_dev], axis=0).drop(['index'], axis=1)
    test_set = pd.concat([trojans_test, clean_test], axis=0).drop(['index'], axis=1)

    print("train_dev_test_set: Done!")

    return train_set, dev_set, test_set


# ######################################################### OLDOLDOLDOLDOLDOLDOLDOLDOLDOLDOLDOLDOLDOLDOLDOLD

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


