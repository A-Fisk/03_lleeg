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
stage_list = ["NR", "N1", "R", "R1"]
sleep_count_df = prep._sum_dataframe(data_list=df_list,
                                     stage_list=stage_list)

### HACK AS HAVEN'T FINISHED SCORING
sleep_count_df.iloc[4, -1] = 12000

# Step three: convert to hours
sleep_hours = prep.convert_to_units(sleep_count_df,
                                    base_freq= "4S",
                                    target_freq="1H")

# Step four: Plot as a scatter/box plot
fig, ax = plot._total_plot(sleep_hours,
                           figsize=(10,10))

# save the damn figure!
save_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/01_projects/'
                        'P3_LLEEG_Chapter3/03_analysis_outputs/04_amount_sleep')
save_plot = save_dir / "01_total_sleep.png"
plt.savefig(save_plot)


dark = [x.between_time("00:00:00", "12:00:00") for x in df_list]
light = [x.between_time("12:00:00", "00:00:00") for x in df_list]
for dark_df, light_df, df in zip(dark, light, df_list):
    name = df.name
    dark_df.name = name
    light_df.name = name
    
time_of_day_dict = {}
time_of_day_dict["dark"] = prep._sum_dataframe(dark, stage_list)
time_of_day_dict["light"] = prep._sum_dataframe(light, stage_list)

time_of_day_df = pd.concat(time_of_day_dict)

# convert to long form data
data = time_of_day_df.stack().reset_index()

fig, ax = plt.subplots()

sns.swarmplot(data=data, x='level_2', y=0, hue="level_0", ax=ax,
              color='0.2', dodge=True)
sns.boxplot(data=data, x='level_2', y=0, hue="level_0", ax=ax,
            fliersize=0)

plot._total_plot(time_of_day_df)



