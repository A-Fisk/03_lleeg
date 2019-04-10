# Script for plotting PIR actogram

import pathlib
import pandas as pd
idx = pd.IndexSlice
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.dates as mdates
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                "07_python_package/actiPy")
import actiPy.actogram_plot as aplot
import actiPy.preprocessing as prep
import actiPy.periodogram as per

fig_dir = pathlib.Path(
    "/Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/"
    "03_lleeg/03_analysis_outputs/05_figures/01_fig1"
)

# Import data
index_cols = [0]
pir_dir = pathlib.Path(
    "/Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/03_lleeg"
    "/01_data_files/10_pirfiles"
)
file_names = sorted(pir_dir.glob("*.csv"))
stem_names = [x.stem for x in file_names]
dfs = [pd.read_csv(
    x,
    parse_dates=True,
    index_col=index_cols
) for x in file_names]
col_names = ["PIR1", "PIR2", "PIR3", "PIR4", "Lights0", "LDR"]
for df in dfs:
    df.drop(
        [df.columns[0], df.columns[-1], df.columns[-3]],
        axis=1,
        inplace=True
    )
    df.columns = col_names
    df.dropna(inplace=True)
curr_df = dfs[1]
both_dict = dict(zip(stem_names, dfs))
both_df = pd.concat(both_dict, sort=False)


# When does the light turn on ?
ldr_label = curr_df.columns[-1]
curr_df.plot(subplots=True)
mask = curr_df.iloc[:, -1] > 100
lights_on_first_day = curr_df[mask].loc["2018-04-10"]
time_on = lights_on_first_day.first_valid_index()
# Shift index
new_lights = curr_df.loc[time_on:, :].copy()
new_lights[ldr_label] = 0

# Fix LDR
new_lights.loc["2018-04-10 08:37": "2018-04-10 20:37", ldr_label] = 500
new_lights.loc["2018-04-11 08:37": "2018-04-25 20:37", ldr_label] = 500
new_lights.loc["2018-04-26 08:37": "2018-04-26 20:37", ldr_label] = 500
new_lights.loc["2018-04-27 08:37": "2018-04-27 20:37", ldr_label] = 500

# Calculate period and save
ll_data = both_df.loc[idx[:, "2018-04-11 08:37":"2018-04-25 08:37"], :].drop(
    ldr_label, axis=1)
periods = ll_data.groupby(
    level=0
).apply(
    per.get_period,
    return_periods=True,
    return_power=False,
    drop_lastcol=False
)
relabel_periods = periods.groupby(
    level=0
).apply(
    prep.label_anim_cols
).stack().reset_index(
    level=0,
    drop=True
)
relabel_df = pd.DataFrame(relabel_periods).T
mean_periods = per.get_secs_mean_df(relabel_df)
mean_savename = fig_dir / "02_periods.csv"
mean_periods.to_csv(mean_savename)

# Plot as actogram
fig, ax = aplot._actogram_plot_from_df(
    new_lights,
    animal_number=2,
    drop_level=False,
    fname=file_names[1],
    day_label_size=10,
    xlabel="Time, ZT",
)
fig.subplots_adjust(hspace=0)
a4 = [8.27, 11.69]
quarter = [x/2 for x in a4]
fig.set_size_inches(quarter[0], quarter[1])
# Save actogram
save_fig = fig_dir / "03_actogram.png"
plt.savefig(save_fig, dpi=600)

plt.close('all')


# Plot as actogram
fig, ax = aplot._actogram_plot_from_df(
    new_lights,
    animal_number=4,
    drop_level=False,
    fname=file_names[1],
    day_label_size=10,
    xlabel="Time, ZT",
)
fig.subplots_adjust(hspace=0)
a4 = [8.27, 11.69]
quarter = [x/2 for x in a4]
fig.set_size_inches(quarter[0], quarter[1])
save_fig = fig_dir / "05_actogram_lights.png"
plt.savefig(save_fig, dpi=600, transparent=True)

# Save actogram


