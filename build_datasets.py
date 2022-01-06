import pandas as pd
import numpy as np
from utils import *

##########################################################
# Set parameters.

ht_length = 1000  # HT points per instance
total_num_of_HT_rows = 100000  # total number of HT rows to be generated
ht_count = int(total_num_of_HT_rows/ht_length)   # number of HT locations
dev_test_ratio = (0.7, 0.3)
averaging_level = 5

ht_distribution = {
    "mean": 10,                 # normal distribution mean
    "sigma": 1,                 # normal distribution standard deviation
    "min": 10,                  # uniform distribution min
    "max": 15,                  # uniform distribution max
}

ht_params_dictionary = {
    "ht_count": ht_count,
    "ht_length": ht_length,  # If you later decide to have several values in a list, sort ht_length from large to small!
    "ht_column_choice": None,
    "ht_distribution": ht_distribution,
    "ht_distribution_type": "normal",                   # 'normal' or 'uniform'
}

# ht_lengths = [20, 50, 100] # In case of several values in this list, ht_length should be sorted from large to small!
# ht_params_dictionary = {
#     "ht_count": ht_count,
#     "ht_lengths": ht_lengths,
#     "ht_column_choice": None,
#     "ht_distribution_tuples": [("normal", 10, 1), ("normal", 15, 1), ("normal", 20, 1)],
# }
#


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


##########################################################
# Calculate the moving averages data frame.

df = moving_average_panda(ht_infected_dataset, avg_lvl=averaging_level, drop_initial_data=False)


##########################################################
# Create Train, Development and Test sets.

_1, _2, ht_affected_indices_all = cache
train_set, dev_set, test_set = train_dev_test_set(df, dev_test_ratio, ht_affected_indices_all)


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
          "mean = {}\nst. dev. = {}\n".format(ht_distribution["mean"], ht_distribution["sigma"]),
          "min = {}\nmax = {}\n".format(ht_distribution["min"], ht_distribution["max"]), file=text_file)


"""
# np.savetxt("C:/Users/Gor/Desktop/PhD_project_2/Important_files-3/new_DATA/Split_into_train_test/HT_toDraw2Ddiagram/4/my_training_data.txt", training_data, fmt='%.2f', delimiter=", ")
# np.savetxt("C:/Users/Gor/Desktop/PhD_project_2/Important_files-3/new_DATA/Split_into_train_test/HT_toDraw2Ddiagram/4/my_testing_data.txt", testing_data, fmt='%.2f', delimiter=", ")
# np.savetxt("C:/Users/Gor/Desktop/PhD_project_2/Important_files-3/new_DATA/Split_into_train_test/HT_toDraw2Ddiagram/4/my_trojan_data.txt", trojan_rows, fmt='%.2f', delimiter=", ")
# with open( "C:/Users/Gor/Desktop/PhD_project_2/Important_files-3/new_DATA/Split_into_train_test/HT_toDraw2Ddiagram/4/README.txt", "w") as text_file:
#     print("The following parameters have been chosen to generate a {} row long HT dataset\n".format(num_of_ht_rows*N),
#           "\nAveraging level = {}\nHT points = {}\nHT P_min = {}\nHT P_max = {}".format(averaging_level, num_of_ht_rows, P_trojan_min, P_trojan_max), file=text_file)

"""


print("Datasets successfully saved.", "Code complete!!!", sep="\n")
