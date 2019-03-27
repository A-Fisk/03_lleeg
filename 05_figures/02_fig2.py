# Script for making figure 2
# Left column - Baseline Day 0, Day 1 of LL7 Delta hypnogram
# Right column group total sleep, NREM, REM, Wake? - per 24 hours

# imports
import pandas as pd
import numpy as np
import pingouin as pg
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
                       "02_fig2.png")

# Step 1 Import the files
file_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/01_projects/'
                        '01_thesisdata/03_lleeg/01_data_files'
                        '/07_clean_fft_files')
file_names = sorted(file_dir.glob("*7*.csv"))
df_list = [prep.read_file_to_df(x, index_col=INDEX_COLS) for x in file_names]

# turn into a dict
df_names = [x.name for x in df_list]
df_dict = dict(zip(df_names, df_list))

spectrum_df = pd.concat(df_dict)
spectrum_df = spectrum_df.loc[idx[:, :"LL_day2", :], :]

#same thing with stage df
stage_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/01_projects/'
                        '01_thesisdata/03_lleeg/01_data_files'
                        '/08_stage_csv')
stage_names = sorted(stage_dir.glob("*.csv"))
stage_list = [prep.read_file_to_df(x, index_col=[0]) for x in
              stage_names]
stage_dfnames = [x.name for x in stage_list]
stage_dict = dict(zip(stage_dfnames, stage_list))
stage_df = pd.concat(stage_dict)
stage_df = stage_df.loc[:, :"LL_day2"]

# Step 2 create delta power file
band = ["Delta"]
range_to_sum = ("0.50Hz", "4.00Hz")
delta_df = prep.create_df_for_single_band(spectrum_df, name_of_band=band,
                                         range_to_sum=range_to_sum)

# Step 3 Count sleep, NREM, REM and wake
sleep_stages = ["NR", "N1", "R", "R1"]
nrem_stages = ["NR", "N1"]
rem_stages = ["R", "R1"]
wake_stages = ["W", "W1", "M"]

unstacked_stages = stage_df.unstack(level=0)
unstacked_stages.name = "stages"

total_sleep = prep.lightdark_df(unstacked_stages, stage_list=sleep_stages,
                                data_list=False)
nrem_sleep = prep.lightdark_df(unstacked_stages, stage_list=nrem_stages,
                                data_list=False)
rem_sleep = prep.lightdark_df(unstacked_stages, stage_list=rem_stages,
                               data_list=False)
wake_count = prep.lightdark_df(unstacked_stages, stage_list=wake_stages,
                               data_list=False)

totals_dict = {
    "Sleep": total_sleep,
    "NREM": nrem_sleep,
    "REM": rem_sleep,
    "Wake": wake_count
}

totals_df = pd.concat(totals_dict)

# Stats ########################################################################

# 1. Does LL affect the time in each stage?
# Repeated Measures 1 way anova for each stage type.
# Not doing 2 way as the markers are all inter-related

save_test_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                             "01_projects/01_thesisdata/03_lleeg/"
                             "03_analysis_outputs/05_figures/00_csvs/02_fig2")

# grab values
stat_colnames = ["Day", "Animal", "Time", "Value"]
dep_var = stat_colnames[-1]
day = stat_colnames[0]
anim = stat_colnames[1]
time = stat_colnames[2]
time_vals = total_sleep.columns

# rm for each tim val
for part in time_vals:
    print(part)
    part_dir = save_test_dir / part
    # perform rm anova for each stage type
    for key, df in zip(totals_dict.keys(), totals_dict.values()):
        
        print(key)
        
        # tidy data
        long_df = df.stack().reset_index()
        long_df.columns = stat_colnames
        part_df = long_df.query("%s == '%s'"%(time, part))
        
        # do anova
        part_rm = pg.rm_anova(dv=dep_var,
                              within=day,
                              subject=anim,
                              data=part_df)
        pg.print_table(part_rm)

        # do posthoc
        ph  = pg.pairwise_tukey(dv=dep_var,
                                between=day,
                                data=part_df)
        pg.print_table(ph)
        
        stage_test_dir = part_dir / key
        anova_file = stage_test_dir / "01_anova.csv"
        ph_file = stage_test_dir / "02_posthoc.csv"
        
        part_rm.to_csv(anova_file)
        ph.to_csv(ph_file)

################################################################################
# Plotting #
fig = plt.figure()

# Step 4 Plot Delta hypnogram left column
hypnogram_grid = gs.GridSpec(nrows=3, ncols=1, figure=fig,
                             right=0.45)

# plot each day of the hypnogram df of LL7 on separate axes
# create the axes and grab into a list
day_axes = []
for row in range(3):
    add_ax = plt.subplot(hypnogram_grid[row])
    day_axes.append(add_ax)

days = spectrum_df.index.get_level_values(1).unique()
ders = spectrum_df.index.get_level_values(2).unique()
label_col = -1
max_delta = delta_df.max()[0]

# on a single axis plot a single days delta
for day, curr_ax in zip(days, day_axes):
    day_data = delta_df.loc[idx[:, day, ders[0]], :]
    
    # remove artefacts and separate into stages to loop through
    stages_list = plot._create_plot_ready_stages_list(day_data)

    # plot each stage one by one on the axis
    for stage_data in stages_list:
        # resample to remove interpolation lines
        curr_data = stage_data.resample(BASE_FREQ, level=3).mean()
        
        label = stage_data.iloc[0, label_col]
        curr_ax.plot(curr_data,
                     label=label)
     
    # TODO fill in grey background
    dark_index = curr_data.between_time("12:00:00", "23:59:00").index
    if day == days[0]:
        alpha = 0.3
    else:
        alpha = 0.1
    curr_ax.fill_between(dark_index, max_delta, 0,
                         facecolors='k', alpha=alpha)
    
    # TODO xlims
    min_time = '2018-01-01 00:00:00'
    max_time = '2018-01-01 23:59:59'
    # TODO ylims
    max_ylim = 60000
    # TODO ylabels
    ylabel = "Delta power, Mv2 (?)"
    curr_ax.set(xlim=[min_time, max_time],
                ylim=[0, max_ylim],
                title=day,
                ylabel=ylabel)
    
    # TODO legend
    if day == days[0]:
        curr_ax.legend(loc=2)
    
    # TODO format dates
    xfmt = mdates.DateFormatter("%H:%M:%S")
    curr_ax.xaxis.set_major_formatter(xfmt)
fig.autofmt_xdate()

curr_ax.set_xlabel("Time of Day, ZT hours")
    
    

# Step 5 Plot totals right hand side - use seaborn
# create the axes to plot on
totals_grid = gs.GridSpec(nrows=4, ncols=1, figure=fig,
                          left=0.55, hspace=0.1)

# create an axis for each total to plot
totals_axes = []
for row in totals_grid:
    add_ax = plt.subplot(row)
    totals_axes.append(add_ax)
    
# totals plotting constants
plotting_columns = ["Day", "Animal", "Light", "Sum"]
day_col = plotting_columns[0]
anim = plotting_columns[1]
lights = plotting_columns[2]
sum_data = plotting_columns[3]
order = ["total", "dark", "light"]

# for each count type
for count_label, curr_total_ax in zip(totals_dict.keys(), totals_axes):

    count_data = totals_dict[count_label]
    # tidy up the data
    # change units to s
    converted_count = ((count_data * 4) / 60) / 60
    totals_plotting_data = converted_count.stack().reset_index()
    totals_plotting_data.columns = plotting_columns

    # plot data as a boxplot with individual points
    sns.boxplot(x=day_col, y=sum_data, hue=lights, data=totals_plotting_data,
                ax=curr_total_ax, hue_order=order, fliersize=0)

    sns.stripplot(x=day_col, y=sum_data, hue=lights, data=totals_plotting_data,
                ax=curr_total_ax, dodge=True, hue_order=order)

    # modify the legend
    ax_leg = curr_total_ax.legend()
    ax_handles, ax_labels = curr_total_ax.get_legend_handles_labels()
    ax_leg.remove()
    
    # separate the days by a line
    curr_total_ax.axvline(0.5, color="k", alpha=0.5)
    curr_total_ax.axvline(1.5, color="k", alpha=0.5)
    
    # TODO set ylims
    # TODO set y titles
    ylabel = "Hours of stage"
    # set REM ymax
    ymax = 16
    if count_label == "REM":
        ymax = 3
    curr_total_ax.set(ylim=[0, ymax],
                      ylabel=ylabel)
    
    # set the title for the column
    if curr_total_ax == totals_axes[0]:
        curr_total_ax.set_title("Time in stages per 24 hours")
        
    curr_total_ax.text(1.0, 0.5,
                       count_label, transform=curr_total_ax.transAxes,
                       rotation=270)

# put the legend for the entire right column
fig.legend(handles=ax_handles[:3], loc=(0.9, 0.8))

# set the figure title
fig.suptitle("Constant light increases sleep in the first 24 hours of "
             "exposure")
fig.set_size_inches(11.69, 8.27)

plt.savefig(SAVEFIG, dpi=600)

# TODO
# Totals
# TODO black line around striplot


plt.close('all')
