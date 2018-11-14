# script to run to check remove header working as expected
import pathlib
import sys
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/sleepPy")
import sleepPy.preprocessing as prep
import sleepPy.plots as plot
import seaborn as sns
sns.set()

# define import dir
input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects"
                         "/P3_LLEEG_Chapter3/01_data_files/07_clean_fft"
                         "/01_reindexed")

# file_list = sorted(input_dir.glob("*.csv"))

kwargs = {"index_col":0,
          "header":0,
          "check_cols":False,
          "rename_cols":False,
          "drop_cols":False}

file_dict = prep.get_all_files_per_animal(input_dir,
                                          single_der=False)

key = list(file_list.keys())[0]

file_list = file_dict[key]


# sleep_stages = ["R","NR","NR1","R1"]
#
# df_dict = {}
#
# for file in file_list:
#     df = prep.read_file_to_df(file,
#                               **kwargs)
#
#     filter = df.iloc[:,0].isin(sleep_stages)
#     df_filt = df.where(filter)
#
#     delta_power = prep.create_df_for_single_band(df_filt,
#                                                  ["Delta"],
#                                                  ("0.50Hz", "4.00Hz"))
#
#     delta_cumsum = delta_power.iloc[:,0].cumsum()
#     df_dict[file] = delta_cumsum
#
# day_one = df_dict[file_list[9]]
# day_two = df_dict[file_list[3]]
# day_three = df_dict[file_list[12]]
#
# fig, ax = plt.subplots()
#
#
# #
# # ax.plot(day_one, "b", label=file_list[9].stem)
# # # ax.plot(day_two, "g", label=file_list[3].stem)
# # ax.plot(day_three, "r", label=file_list[12].stem)
#
# fig.legend()