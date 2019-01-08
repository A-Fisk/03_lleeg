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
                        "/01_thesisdata/03_lleeg/01_data_files/08_stage_csv")
file_list = sorted(input_dir.glob("*.csv"))

read_kwargs = {"index_col": [0],
               "header": [0]}
df_list = [prep.read_file_to_df(x, **read_kwargs)
           for x in file_list]

# Define the variables that will change
stage_list = ["W", "W1"]
label = "time awake, (hours)"
title = "Time awake per day"
save_name = "04_wake.png"

# Step two: Count the sleep
sleep_count_df = prep.lightdark_df(df_list=df_list,
                                   stage_list=stage_list)

### HACK AS HAVEN'T FINISHED SCORING
# sleep_count_df.iloc[4, -1] = 12000

# Step three: convert to hours
sleep_hours = prep.convert_to_units(sleep_count_df,
                                    base_freq= "4S",
                                    target_freq="1H")
# convert to long form data
data = sleep_hours.stack().reset_index()
x = "Experimental_day"
y = label
hue = "Light"
cols = [hue, "Animal", x, label]
data.columns = cols

# Step four: Plot as a scatter/box plot
fig, ax = plot.dark_light_plot(data,
                               x=x, y=y, hue=hue,
                               figsize=(10,10),
                               title=title)

# save the damn figure!
save_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/01_projects/'
                        '01_thesisdata/03_lleeg/03_analysis_outputs'
                        '/04_amounts')
save_plot = save_dir / save_name
plt.savefig(save_plot)


