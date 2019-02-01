# Figure 4 script

# Comparing the spectrum in zt 12-24 for the first two days in LL
# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.dates as mdates
import seaborn as sns
sns.set()
import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/sleepPy")
import sleepPy.preprocessing as prep
import sleepPy.plots as plot

# define constants
INDEX_COLS = [0, 1, 2]
idx = pd.IndexSlice
BASE_FREQ = "4S"
SAVEFIG = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects/"
                       "01_thesisdata/03_lleeg/03_analysis_outputs/05_figures/"
                       "04_fig4.png")

# Step 1 Import files and tidy
# need both stage csvs and spectral
file_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/01_projects/'
                        '01_thesisdata/03_lleeg/01_data_files'
                        '/07_clean_fft_files')
file_names = sorted(file_dir.glob("*.csv"))
df_list = [prep.read_file_to_df(x, index_col=INDEX_COLS) for x in file_names]

# turn into a dict
df_names = [x.name for x in df_list]
df_dict = dict(zip(df_names, df_list))

spectrum_df = pd.concat(df_dict)
spectrum_df = spectrum_df.sort_index()

# Step 2 select just the right ZTs
zt_slice = spectrum_df.loc[idx[:, :, :, "2018-01-01 12:00:00":], :]

# Step 3 Select just the state of interest - wake and NREM
wake_mask = zt_slice["Stage"] == "W"
nrem_mask = zt_slice["Stage"] == "NR"

wake_slice = zt_slice.where(wake_mask).drop("Stage", axis=1)
nrem_slice = zt_slice.where(nrem_mask).drop("Stage", axis=1)

# Step 4 Normalise each animal to it's own baseline day
# get the mean spectrum for each day for each animal.
wake_spectrum = wake_slice.groupby(level=[0, 1, 2]).mean()
nrem_spectrum = nrem_slice.groupby(level=[0, 1, 2]).mean()

# remove FOC derivation for now
for df in wake_spectrum, nrem_spectrum:
    df.drop("foc", level=2, inplace=True)
    
# normalise to each animals baseline
def normalise_to_baseline(data):
    """
    Normalised everything to the first level of the second level of the
    multi-index
    """
    days = data.index.get_level_values(1).unique()
    animals = data.index.get_level_values(0).unique()
    data_list = []
    for animal in animals:
        baseline = data.loc[idx[animal, days[0], :, :], :].values
        day_list = []
        for day in days:
            if animal == "LL6" and day == days[-1]:
                continue
            exp_day = data.loc[idx[animal, day, :, :], :]
            normalised = exp_day.sub(baseline, level=1)
            day_list.append(normalised)
        day_df = pd.concat(day_list)
        data_list.append(day_df)
    normalised_data = pd.concat(data_list)
    
    return normalised_data

normalised_wake = normalise_to_baseline(wake_spectrum)
normalised_nrem = normalise_to_baseline(nrem_spectrum)

# Step 5 turn into longform data for plotting
long_wake = normalised_wake.stack().reset_index()
long_nrem = normalised_nrem.stack().reset_index()

cols = ["Animal", "Light_period", "Derivation", "Frequency", "Power"]
long_wake.columns = cols
long_nrem.columns = cols

# Step 6 Plot wake on top two and nrem on bottom two subplots

# tidy data further - remove LL1 and baselinse period
baseline_wake_mask = long_wake["Light_period"] == "Baseline_-0"
long_wake = long_wake.mask(baseline_wake_mask)
ll1_wake_mask = long_wake["Animal"] == "LL1"
long_wake = long_wake.mask(ll1_wake_mask)

baseline_nrem_mask = long_nrem["Light_period"] == "Baseline_-0"
long_nrem = long_nrem.mask(baseline_nrem_mask)
ll1_nrem_mask = long_nrem['Animal'] == "LL1"
long_nrem = long_nrem.mask(ll1_nrem_mask)

# Initialise figure
fig = plt.figure()

# figure constants
animals = cols[0]
lights = cols[1]
ders = cols[2]
freq = cols[3]
power = cols[4]
derivations = long_wake[ders].unique()[1:]
capsize = 1.5
scale = 0.8
linesize = 2
markersize = 0.2
alpha = 0.5
labelsize = 7.5
xticks = np.linspace(0, 80, 21)
xticklabels = long_wake[freq].unique()[1::4]
ylabel = "% change from Baseline day"
hline_kwargs = {"linestyle": "--",
                "color": "k"}

# create top two subplots for wake
upper_plots = gs.GridSpec(nrows=1, ncols=2, figure=fig, bottom=0.55)

upper_axes = []
for plot in upper_plots:
    add_ax = plt.subplot(plot)
    upper_axes.append(add_ax)
    
# wake constants
wake_ymin = -25
wake_ymax = 40
    
# plot the separate derivation on each plot
for der, curr_top_ax in zip(derivations, upper_axes):
    
    curr_der_mask = long_wake[ders] == der
    curr_der_data = long_wake.where(curr_der_mask)
    
    sns.pointplot(x=freq, y=power, hue=lights, data=curr_der_data, ci=68,
                  ax=curr_top_ax, capsize=capsize,
                  errwidth=linesize, scale=scale)
    
    curr_legend = curr_top_ax.legend()
    if der != derivations[-1]:
        curr_legend.remove()
        
    # rotation and shrink the x axis
    curr_top_ax.set_xticks(xticks)
    curr_top_ax.set_xticklabels(xticklabels,
                                rotation=30, ha='right', size=labelsize)
    
    # set axis values
    curr_top_ax.set(ylim=[wake_ymin, wake_ymax],
                    title=der,
                    ylabel=ylabel)

    # set properties
    plt.setp(curr_top_ax.collections, facecolors='none')
    plt.setp(curr_top_ax.get_lines(), linewidth=linesize)
    
    # set line at 0
    curr_top_ax.axhline(0, **hline_kwargs)
    
    # turn on gridlines vertical
    curr_top_ax.xaxis.grid(True)
    
# set the type of spectrum on RHS
curr_top_ax.text(1.0, 0.5, "Wake", transform=curr_top_ax.transAxes,
                 rotation=270)


# create bottom two subplots for nrem
lower_plots = gs.GridSpec(nrows=1, ncols=2, figure=fig, top=0.45)

lower_axes = []
for plot in lower_plots:
    add_ax = plt.subplot(plot)
    lower_axes.append(add_ax)
    
# nrem constants
nrem_ymin = -60
nrem_ymax = 40
    
# plot the separate derivation on each plot
for der, curr_bottom_ax in zip(derivations, lower_axes):
    
    curr_der_mask = long_nrem[ders] == der
    curr_der_data = long_nrem.where(curr_der_mask)
    
    sns.pointplot(x=freq, y=power, hue=lights, data=curr_der_data, ci=68,
                  ax=curr_bottom_ax, capsize=capsize,
                  errwidth=linesize, scale=scale)

    curr_legend = curr_bottom_ax.legend()
    curr_legend.remove()

    curr_bottom_ax.set_xticks(xticks)
    curr_bottom_ax.set_xticklabels(xticklabels,
                                   rotation=30, ha='right', size=labelsize)
    
    # set limits
    curr_bottom_ax.set(ylim=[nrem_ymin, nrem_ymax],
                       ylabel=ylabel)

    plt.setp(curr_bottom_ax.collections, facecolors='none')
    plt.setp(curr_bottom_ax.get_lines(), linewidth=linesize)
    
    # set line at 0
    curr_bottom_ax.axhline(0, **hline_kwargs)

    # turn on gridlines vertical
    curr_bottom_ax.xaxis.grid(True)
    
# set type
curr_bottom_ax.text(1.0, 0.5, "NREM", transform=curr_bottom_ax.transAxes,
                    rotation=270)

fig.suptitle("Constant light changes spectral power of wake and sleep during "
             "ZT "
             "12-24")

fig.set_size_inches(11.69, 8.27)

plt.savefig(SAVEFIG, dpi=600)

plt.close('all')
