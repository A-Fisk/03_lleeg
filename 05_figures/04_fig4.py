# Figure 4 script
# Comparing the spectrum in zt 12-18 in the first two days in LL

# imports
import pandas as pd
import numpy as np
import pingouin as pg
import matplotlib
#matplotlib.use('macosx')
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
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/actiPy")
import actiPy.episodes as ep
import actiPy.preprocessing as aprep

# define constants
INDEX_COLS = [0, 1, 2]
idx = pd.IndexSlice
BASE_FREQ = "4S"
SAVEFIG = pathlib.Path(
    "/Users/angusfisk/Documents/01_PhD_files/01_projects/"
    "01_thesisdata/03_lleeg/03_analysis_outputs/05_figures/"
    "04_fig4.png"
)
OFFSET = pd.Timedelta("30M")

# Step 1 Import files and tidy
# need both stage csvs and spectral
file_dir = pathlib.Path(
    '/Users/angusfisk/Documents/01_PhD_files/01_projects/'
    '01_thesisdata/03_lleeg/01_data_files'
    '/07_clean_fft_files'
)
file_names = sorted(file_dir.glob("*.csv"))
df_list = [prep.read_file_to_df(x,
                                parse_dates=True,
                                index_col=INDEX_COLS)
           for x in file_names]
df_names = [x.name for x in df_list]
df_dict = dict(zip(df_names, df_list))
spectrum_df = pd.concat(df_dict)
spectrum_df = spectrum_df.sort_index()
spectrum_df = spectrum_df.loc[idx["LL2":, :"LL_day2", :], :]

# same with stage df
stage_dir = pathlib.Path(
    '/Users/angusfisk/Documents/01_PhD_files/01_projects/'
    '01_thesisdata/03_lleeg/01_data_files'
    '/08_stage_csv'
)
stage_names = sorted(stage_dir.glob("*.csv"))
stage_list = [
    prep.read_file_to_df(x,
                         parse_dates=True,
                         index_col=[0]) for x in stage_names ]
stage_dfnames = [x.name for x in stage_list]
stage_dict = dict(zip(stage_dfnames, stage_list))
stage_df = pd.concat(stage_dict)
stage_df = stage_df.loc[:, :"LL_day2"]

# Step 2 select just the right ZTs
der = 'fro'
zt_slice = spectrum_df.loc[
           idx["LL2":, :, der, "2018-01-01 12:00:00":"2018-01-01 17:00:00"],
           : # removes LL2
           ]

# Step 3 Select just the state of interest - wake and NREM
nrem_mask = zt_slice["Stage"] == "NR"
nrem_slice = zt_slice.where(nrem_mask).drop("Stage", axis=1)
nrem_mean = nrem_slice.groupby(level=[0, 1, 2]).mean()
nrem_mean_mean = nrem_mean.groupby(level=1).mean()
nrem_mean_sem = nrem_mean.groupby(level=1).sem()
long_nrem = nrem_mean_mean.iloc[:, 3:].stack().reset_index()
cols = ["Animal", "Light_period", "Derivation", "Frequency", "Power"]
long_nrem_cols = [cols[x] for x in [1, 3, 4]]
long_nrem.columns = long_nrem_cols
long_nrem["SEM"] = nrem_mean_sem.iloc[:, 3:].stack().values

# log transform to normalise for stats
# log_nrem = long_nrem.copy()
# log_nrem_vals = np.log10(log_nrem.iloc[:, -2:])
# log_nrem.iloc[:, -2:] = log_nrem_vals

# Step 4 Calculate episodes of activity per hour

# Only code NR episodes (NR and NR1)
nr_stages = ["NR", "N1", "M"]
nr_int_df = stage_df.isin(nr_stages).astype(int)
nr_episodes_df = nr_int_df.groupby(
    level=0
).apply(
    ep.episode_find_df,
    drop_level=True,
    reset_level=False,
    remove_lights=False
)

# Use actiPy episode finder to get lengths and times of episodes
# same for wake
wake_stages = ["W", "W1"]
wake_int_df = stage_df.isin(wake_stages).astype(int)
wake_episodes_df = wake_int_df.groupby(
    level=0
).apply(
    ep.episode_find_df,
    drop_level=True,
    reset_level=False,
    remove_lights=False
)

# Get the number of episodes using sum resample
nr_ep_mask = (nr_episodes_df > 1).astype(bool)
nr_ep_sum = nr_ep_mask.groupby(
    level=0
).resample(
    "H",
    level=1,
    loffset=OFFSET
).sum().loc[idx[:, "2018-01-01 00:30":"2018-01-01 23:30"], :]
long_sum = nr_ep_sum.stack().reset_index()

# get the mean duration using mean
nr_ep_mean = nr_episodes_df.groupby(
    level=0
).resample(
    "H",
    level=1,
    loffset=OFFSET
).mean() / 60
nr_ep_mean_slice = nr_ep_mean.loc[
                   idx[:, "2018-01-01 00:30":"2018-01-01 23:30"], :]
long_mean = nr_ep_mean_slice.stack().reset_index()
long_frag = long_sum.copy()
frag_cols = [
    "Animal",
    "Time",
    "Day",
    "Count"
]
long_frag.columns = frag_cols
long_frag["Mean"] = long_mean.iloc[:, -1]


# Stats ########################################################################

save_test_dir = pathlib.Path(
    "/Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata"
    "/03_lleeg/03_analysis_outputs/05_figures/00_csvs/04_fig4"
)

# Q1 Does power of the spectrum change between days
# Repeated two way ANOVA of Power ~ Freq*days | anim

stats_spec_df = np.log10(nrem_mean)
stats_spec_df.index = stats_spec_df.index.droplevel(2)
stats_spec_df = stats_spec_df.stack().reset_index()

anim_col = stats_spec_df.columns[0]
day_col = stats_spec_df.columns[1]
freq_col = stats_spec_df.columns[2]
power_col = stats_spec_df.columns[3]

spec_rm = pg.mixed_anova(
    dv=power_col,
    within=day_col,
    between=freq_col,
    subject=anim_col,
    data=stats_spec_df
)
pg.print_table(spec_rm)

spec_name = save_test_dir / "01_spec_anova.csv"
spec_rm.to_csv(spec_name)

# Q2 Does the Number of episodes change between day?
# Rpeated two way anova of Count ~ Time*day | anim

count_stats_df = long_frag.copy()

anim_col = count_stats_df.columns[0]
time_col = count_stats_df.columns[1]
day_col = count_stats_df.columns[2]
count_col = count_stats_df.columns[3]
mean_col = count_stats_df.columns[4]

count_rm = pg.rm_anova2(
    dv=count_col,
    within=[day_col, time_col],
    subject=anim_col,
    data=count_stats_df
)
pg.print_table(count_rm)

count_rm_name = save_test_dir / "02_count_rm.csv"
count_rm.to_csv(count_rm_name)

count_ph = aprep.tukey_pairwise_ph(
    count_stats_df,
    hour_col=time_col,
    dep_var=count_col,
    protocol_col=day_col
)
count_ph_name = save_test_dir / "02_count_ph.csv"
count_ph.to_csv(count_ph_name)

# Q3 Does duration of episodes change between day?
# repeated two way anova of duration ~ Time*day | anim

mean_rm = pg.rm_anova2(
    dv=mean_col,
    within=[day_col, time_col],
    subject=anim_col,
    data=count_stats_df
)
pg.print_table(mean_rm)
mean_name = save_test_dir / "03_mean_rm.csv"
mean_rm.to_csv(mean_name)

mean_ph = aprep.tukey_pairwise_ph(
    count_stats_df,
    hour_col=time_col,
    dep_var=mean_col,
    protocol_col=day_col
)
mean_ph_name = save_test_dir / "03_mean_ph.csv"
mean_ph.to_csv(mean_ph_name)


# Plotting #####################################################################
# figure constants
anim_col = cols[0]
# animals = long_nrem[anim_col].unique()
lights = cols[1]
ders = cols[2]
freq = cols[3]
power = cols[4]
# derivations = long_nrem[ders].unique()[1:]
frag_data_types = ["Count", "Mean"]
time_col = frag_cols[1]
day_col = frag_cols[2]

# tidy/size constants
capsize = 1.5
scale = 0.8
linesize = 2
errwidth = 1
markersize = 0.2
alpha = 0.5
labelsize = 7.5
sem = 68

# extra plot features constants
xfmt = mdates.DateFormatter("%H:%M")
panel_xpos = -0.1
panel_ypos = 1.1



# Initialise figure
fig = plt.figure()

# plot the spectrum on the upper left
spectrum_grid = gs.GridSpec(
    ncols=1,
    nrows=1,
    figure=fig,
    right=0.45,
)
spectrum_axes = [plt.subplot(x) for x in spectrum_grid]

days = long_nrem[lights].unique()
curr_ax_spec = spectrum_axes[0]
for curr_day in days:
    curr_day_data = long_nrem.query("%s == '%s'" %(lights, curr_day))
    
    curr_ax_spec.plot(
        curr_day_data[freq],
        curr_day_data[power]
    )
    curr_ax_spec.fill_between(
        curr_day_data[freq],
        curr_day_data[power] - curr_day_data["SEM"],
        curr_day_data[power] + curr_day_data["SEM"],
        alpha=0.5
    )
    
curr_ax_spec.set_yscale("log")
spec_leg = curr_ax_spec.legend()
spec_leg.remove()
# set x axis to be frequency
spec_xticks = np.linspace(1, 81, 21)
spec_xticklabels = long_nrem[freq].unique()[1::4]
curr_ax_spec.set_xticks(spec_xticks)
curr_ax_spec.set_xticklabels(
    spec_xticklabels,
    rotation=30,
    ha='right',
    size=labelsize
)
curr_ax_spec.set_ylabel(
    "EEG power density, µV$^2$/0.25Hz +/-SEM"
)
curr_ax_spec.set_xlabel(
        "Frequency"
)
fig.text(
    0.5,
    1.05,
    "NREM power spectra",
    transform = curr_ax_spec.transAxes,
    ha='center'
)
# add in panel position
fig.text(
    panel_xpos,
    1.05,
    "A",
    transform=curr_ax_spec.transAxes,
    ha='right'
)


frag_grid = gs.GridSpec(
    nrows=2,
    ncols=1,
    figure=fig,
    hspace=0.1,
    left=0.55,
)
frag_axes = [plt.subplot(x) for x in frag_grid]

frag_xticklabels = long_frag[time_col][::3].dt.strftime("%H:%M")
for type, curr_ax_frag in zip(frag_data_types, frag_axes):
    
    sns.pointplot(
        x=time_col,
        y=type,
        hue=day_col,
        ax=curr_ax_frag,
        data=long_frag,
        capsize=capsize,
        ci=sem,
        errwidth=errwidth
    )
    frag_leg = curr_ax_frag.legend()
    frag_leg.remove()
    curr_ax_frag.set_xticklabels("")
    curr_ax_frag.axvline(
        11.5,
        linestyle="--",
        color='k'
    )
curr_ax_frag.set_xticklabels(
    frag_xticklabels,
    rotation=30,
    ha='right',
    size=labelsize
)
fig.text(
    0.5,
    1.1,
    "NREM Episodes",
    transform=frag_axes[0].transAxes,
    ha='center'
)
frag_labelsize = 10
frag_axes[0].set_ylabel("Number of NREM episodes per hour, \n No. +/-SEM",
        fontsize=frag_labelsize)
frag_axes[0].set_xlabel("")
frag_axes[1].set_ylabel("Mean duration of NREM episodes per hour, \n minutes, +/-SEM",
        fontsize=frag_labelsize)
frag_axes[1].set_xlabel("Time of day, CT hours")

for curr_ax, curr_panel in zip(frag_axes, ["B", "C"]):
    fig.text(
        panel_xpos,
        panel_ypos,
        curr_panel,
        transform=curr_ax.transAxes,
        ha='right'
    )
    
# add stats to count levels
ylevel_day1 = 0.9
ylevel_day2 = 0.95

count_axis = frag_axes[0]
# count_axis.set_ylim([0, 20])

# Get y level
ycoord_day1 = plot.sig_line_coord_get(
    count_axis,
    ylevel_day1
)
ycoord_day2 = plot.sig_line_coord_get(
    count_axis,
    ylevel_day2
)

# get xvals where sig
sig_day1 = [x.strftime("%H:%M") for x in plot.sig_locs_get(count_ph)]
sig_day2 = [x.strftime("%H:%M")
            for x in plot.sig_locs_get(count_ph,index_level2val=1)]

label_loc_dict = plot.get_xtick_dict(frag_axes[1])

plot.draw_sighlines(
    yval=ycoord_day1,
    sig_list=sig_day1,
    label_loc_dict=label_loc_dict,
    minus_val=0.5,
    plus_val=0.5,
    curr_ax=count_axis,
    color="C1"
)
plot.draw_sighlines(
    yval=ycoord_day2,
    sig_list=sig_day2,
    label_loc_dict=label_loc_dict,
    minus_val=0.5,
    plus_val=0.5,
    curr_ax=count_axis,
    color="C1"
)


# same thing for mean a
#
# mean_axis = frag_axes[1]
#
# # Get y level
# ycoord_day1 = plot.sig_line_coord_get(
#     mean_axis,
#     ylevel_day1
# )
# ycoord_day2 = plot.sig_line_coord_get(
#     mean_axis,
#     ylevel_day2
# )
#
# # get xvals where sig
# sig_day1 = [x.strftime("%H:%M") for x in plot.sig_locs_get(mean_ph)]
# sig_day2 = [x.strftime("%H:%M")
#             for x in plot.sig_locs_get(mean_ph,index_level2val=1)]
#
# label_loc_dict = plot.get_xtick_dict(mean_axis)
#
# plot.draw_sighlines(
#     yval=ycoord_day1,
#     sig_list=sig_day1,
#     label_loc_dict=label_loc_dict,
#     minus_val=0.5,
#     plus_val=0.5,
#     curr_ax=mean_axis,
#     color="C1"
# )
# plot.draw_sighlines(
#     yval=ycoord_day2,
#     sig_list=sig_day2,
#     label_loc_dict=label_loc_dict,
#     minus_val=0.5,
#     plus_val=0.5,
#     curr_ax=mean_axis,
#     color="C1"
# )

# figure legend
frag_axes[0].legend(
    loc=(1.02, 0.8),
    fontsize=8,
    markerscale=0.5
)

fig.suptitle("Differences during ZT 12-17")

fig.set_size_inches(11.69, 8.27)
plt.savefig(SAVEFIG, dpi=600)

plt.close('all')
