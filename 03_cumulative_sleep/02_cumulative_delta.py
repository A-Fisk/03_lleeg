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

file_list = sorted(input_dir.glob("*.csv"))

kwargs = {"index_col":0,
          "header":0,
          "check_cols":False,
          "rename_cols":False,
          "drop_cols":False}

df_dict = {}

for file in file_list:
    df = prep.read_file_to_df(file,
                              **kwargs)

    delta_power = prep.create_df_for_single_band(df,
                                                 ["Delta"],
                                                 ("0.50Hz", "4.00Hz"))

    delta_cumsum = delta_power.cumsum()
    df_dict[file] = delta_cumsum

day_one = df_dict[file_list[9]].iloc[:,0]
day_two = df_dict[file_list[3]].iloc[:,0]
day_three = df_dict[file_list[12]].iloc[:,0]

fig, ax = plt.subplots()

ax.plot(day_one, "b", label=file_list[9].stem)
# ax.plot(day_two, "g", label=file_list[3].stem)
ax.plot(day_three, "r", label=file_list[12].stem)

fig.legend()