import pandas as pd
import numpy as np
from utils import *

##########################################################
# Set parameters.
ht_distribution_params = {
    "mean": 10,                 # normal distribution mean
    "sigma": 1,                 # normal distribution standard deviation
    "min": 10,                  # uniform distribution min
    "max": 15,                  # uniform distribution max
}

ht_length = 1000  # HT points per instance
total_num_of_HT_rows = 100000  # total number of HT rows to be generated
ht_count = int(total_num_of_HT_rows/ht_length)   # number of HT locations

ht_params_dictionary = {
    "ht_count": ht_count,
    "ht_length": ht_length,  # If you later decide to have several values in a list, sort ht_length from large to small!
    "ht_column_choice": None,
    "ht_distribution_params": ht_distribution_params,
    "ht_distribution_type": "normal",                   # 'normal' or 'uniform'
}


ht_column = None    # If None, will pick a random available column
averaging_level = 5

P_trojan_min = 10
P_trojan_max = 20

num_of_ht_rows = 1000  # HT points per instance
total_num_of_HT_rows = 100000  # total number of HT rows to be generated
N = int(total_num_of_HT_rows/num_of_ht_rows)   # number of HT locations


just_in_case = 200
buffer_zone = averaging_level + num_of_ht_rows + just_in_case
# ----------


##########################################################
# Load the raw dataset. (Only HT-clean data points)

path = r'../../DATA/newerrrr_DATA/DATA_ready_to_use.txt'
dataset = load_data(path)
"""
# file_path = r'./POWCONS_with_voltages_2.TXT'
# file_path = r'C:/Users/Gor/Desktop/PhD_project_2/Important_files-3/new_DATA/Split_into_train_test/DATA_source/ALL_DATA_MERGED.txt'
# file_path = r'C:/Users/Gor/Desktop/PhD_project_2/Important_files-3/DATA/newerrrr_DATA/DATA_ready_to_use.txt'
file_path = r'../../DATA/newerrrr_DATA/DATA_ready_to_use.txt'
all_data = load_data(file_path)
all_data = all_data[:4000000]
all_data_numpy = np.r_[all_data]
"""

##########################################################
# Add Trojan rows to the data.

ht_infected_dataset, cache = insert_all_trojans(dataset, averaging_level, ht_params_dictionary)

"""
n_all_data_rows, n_all_data_columns = all_data_numpy.shape

initial_available_index_range = np.arange(averaging_level, n_all_data_rows - num_of_ht_rows)
available_index_range = initial_available_index_range
ht_index_list = np.array([], dtype=np.int64)
# Randomly choose N indexes and HT rows
for i in range(N):
    index = np.random.choice(available_index_range)    # choose index
    print()
    print("HT instance: ", i)
    print("HT index range:  ", index, index + num_of_ht_rows)
    unavailable_indices = np.arange(index - buffer_zone, index + buffer_zone + 1)
    available_index_range = np.setdiff1d(available_index_range, unavailable_indices)

    all_data = add_trojan_rows(data_set=all_data, i=index, num_of_trojan_rows=num_of_ht_rows,
                               trojan_min=P_trojan_min, trojan_max=P_trojan_max, ht_column_choice=ht_column)

    ht_index_list = np.append(ht_index_list, np.arange(index, index + num_of_ht_rows))
    ht_affected_index_list = np.append(ht_index_list, np.arange(index, index + num_of_ht_rows + averaging_level))

ht_index_list = np.sort(ht_index_list)      # This list is for later use
print(ht_index_list)

"""
# ----------


##########################################################
# Calculate the moving averages data frame.

df = moving_average_panda(ht_infected_dataset, avg_lvl=averaging_level, drop_initial_data=False)

"""
all_data = moving_average_panda(all_data, averaging_level, drop_initial_data=False)
"""
# ----------

##########################################################
# Create Train, Development and Test sets.

dev_test_ratio = (0.7, 0.3)
_1, _2, ht_affected_indices_all = cache
train_set, dev_set, test_set = train_dev_test_set(df, dev_test_ratio, ht_affected_indices_all)

# ----------

##########################################################
# Separate trojan rows (index:index +  num_of_ht_rows + averaging_level) from clean rows (0:index) and (index +  num_of_ht_rows + averaging_level : end)
# Separate trojan rows from clean rows.
"""
# trojan_free_indexes = np.append(np.arange(averaging_level, index), np.arange(index + num_of_ht_rows + averaging_level, n_all_data_rows))
# trojan_rows = all_data.drop(trojan_free_indexes, axis=0)
# all_data_clean = all_data.drop(range(index, index + num_of_ht_rows + averaging_level), axis=0)
trojan_rows = all_data.iloc[ht_index_list, :]
all_data_clean = all_data.drop(ht_index_list, axis=0)
"""
# ----------

##########################################################
# Split the moving average data frame into train and test.
"""
all_data_clean = all_data_clean.drop_duplicates()
spl_ratio = (all_data_clean.shape[0] - N * num_of_ht_rows)/all_data_clean.shape[0]
training_data, testing_data = split_to_train_test(spl_ratio, all_data_clean)
"""
# ----------

##########################################################
# Append Trojan rows to test data.
"""
#testing_data = np.append(testing_data, trojan_rows, axis=0)
"""
# ----------

##########################################################
# Save data sets in text files

print("Saving datasets...")

np.savetxt("../../DATA/newerrrr_DATA/DELETE_THIS/my_training_set.txt", train_set, fmt='%5.2f', delimiter=", ")
np.savetxt("../../DATA/newerrrr_DATA/DELETE_THIS/my_dev_set.txt", dev_set, fmt='%5.2f', delimiter=", ")
np.savetxt("../../DATA/newerrrr_DATA/DELETE_THIS/my_test_set.txt", test_set, fmt='%5.2f', delimiter=", ")

with open("../../DATA/newerrrr_DATA/DELETE_THIS/README.txt", "w") as text_file:
    print("The following parameters have been chosen to generate a {} row long HT dataset\n".format(ht_length*ht_count),
          "Averaging level = {}\nHT instance length = {}\n\n".format(averaging_level, ht_length),
          "Distribution: {}\n".format(ht_params_dictionary["ht_distribution_type"]),
          "mean = {}\nst. dev. = {}\n".format(ht_distribution_params["mean"], ht_distribution_params["sigma"]),
          "min = {}\nmax = {}\n".format(ht_distribution_params["min"], ht_distribution_params["max"]), file=text_file)


"""
# np.savetxt("C:/Users/Gor/Desktop/PhD_project_2/Important_files-3/new_DATA/Split_into_train_test/HT_toDraw2Ddiagram/4/my_training_data.txt", training_data, fmt='%.2f', delimiter=", ")
# np.savetxt("C:/Users/Gor/Desktop/PhD_project_2/Important_files-3/new_DATA/Split_into_train_test/HT_toDraw2Ddiagram/4/my_testing_data.txt", testing_data, fmt='%.2f', delimiter=", ")
# np.savetxt("C:/Users/Gor/Desktop/PhD_project_2/Important_files-3/new_DATA/Split_into_train_test/HT_toDraw2Ddiagram/4/my_trojan_data.txt", trojan_rows, fmt='%.2f', delimiter=", ")
# with open( "C:/Users/Gor/Desktop/PhD_project_2/Important_files-3/new_DATA/Split_into_train_test/HT_toDraw2Ddiagram/4/README.txt", "w") as text_file:
#     print("The following parameters have been chosen to generate a {} row long HT dataset\n".format(num_of_ht_rows*N),
#           "\nAveraging level = {}\nHT points = {}\nHT P_min = {}\nHT P_max = {}".format(averaging_level, num_of_ht_rows, P_trojan_min, P_trojan_max), file=text_file)


# np.savetxt("C:/Users/Gor/Desktop/PhD_project_2/Important_files-3/DATA/newerrrr_DATA/HT_to_draw_2D_diagrams/my_training_data.txt", training_data, fmt='%.2f', delimiter=", ")
# np.savetxt("C:/Users/Gor/Desktop/PhD_project_2/Important_files-3/DATA/newerrrr_DATA/HT_to_draw_2D_diagrams/my_testing_data.txt", testing_data, fmt='%.2f', delimiter=", ")
# np.savetxt("C:/Users/Gor/Desktop/PhD_project_2/Important_files-3/DATA/newerrrr_DATA/HT_to_draw_2D_diagrams/my_trojan_data.txt", trojan_rows, fmt='%.2f', delimiter=", ")
# with open( "C:/Users/Gor/Desktop/PhD_project_2/Important_files-3/DATA/newerrrr_DATA/HT_to_draw_2D_diagrams/README.txt", "w") as text_file:
#     print("The following parameters have been chosen to generate a {} row long HT dataset\n".format(num_of_ht_rows*N),
#           "\nAveraging level = {}\nHT points = {}\nHT P_min = {}\nHT P_max = {}".format(averaging_level, num_of_ht_rows, P_trojan_min, P_trojan_max), file=text_file)
"""
#----------



print("Done!!!")