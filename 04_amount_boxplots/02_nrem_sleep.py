# Standard Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/sleepPy")
import sleepPy.preprocessing as prep
import sleepPy.plots as plot
import seaborn as sns
sns.set()

# Step one, get the data ready
# import the files into a list
input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects"
                        "/P3_LLEEG_Chapter3/01_data_files/08_stage_csv")
file_list = sorted(input_dir.glob("*.csv"))

read_kwargs = {"index_col": [0],
               "header": [0]}
df_list = [prep.read_file_to_df(x, **read_kwargs)
           for x in file_list]

# Step two: Count the sleep
stage_list = ["NR", "N1"]
sleep_dict = {}
for df in df_list:
    name = df.name
    value = df.isin(stage_list).sum()
    sleep_dict[name] = value
sleep_count_df = pd.concat(sleep_dict).unstack()

### HACK AS HAVEN'T FINISHED SCORING
sleep_count_df.iloc[4, -1] = 12000

# Step three: convert to hours
sleep_hours = prep.convert_to_units(sleep_count_df,
                                    base_freq= "4S",
                                    target_freq="1H")

# Step four: Plot as a scatter/box plot
fig, ax = plt.subplots()

sns.swarmplot(data=sleep_hours, color='k', ax=ax)
sns.boxplot(data=sleep_hours, ax=ax, fliersize=0)

# set the labels
fig.text(0.05,
        0.5,
        "nrem sleep in hours",
        ha="center",
        va="center",
        rotation='vertical')
fig.text(0.5,
         0.03,
         "Experimental Day",
         ha='center',
         va='center')
fig.suptitle("Amount of sleep per day")

# set size for saving
fig.set_size_inches((10,10))

# save the damn figure!
save_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/01_projects/'
                        'P3_LLEEG_Chapter3/03_analysis_outputs/04_amount_sleep')
save_plot = save_dir / "02_nrem_sleep.png"
plt.savefig(save_plot)


