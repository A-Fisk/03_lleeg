# Script for making Figure 2. Delta hypnogram and Time in stages

import pandas as pd
import numpy as np
import pingouin as pg
import matplotlib
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


######### Step 1 Import the files
file_dir = pathlib.Path('/Users/angusfisk/Documents/01_PhD_files/01_projects/'
                        '01_thesisdata/03_lleeg/01_data_files'
                        '/07_clean_fft_files')
file_names = sorted(file_dir.glob("*7*.csv"))
df_list = [prep.read_file_to_df(x, index_col=INDEX_COLS) for x in file_names]
df_names = [x.name for x in df_list]
df_dict = dict(zip(df_names, df_list))
spectrum_df = pd.concat(df_dict)
spectrum_df = spectrum_df.loc[idx[:, :"LL_day2", :], :]

# same thing with stage df
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

########## Step 2 create delta power file
band = ["Delta"]
range_to_sum = ("0.50Hz", "4.00Hz")
delta_df = prep.create_df_for_single_band(
    spectrum_df,
    name_of_band=band,
    range_to_sum=range_to_sum,
    sum=False,
    mean=True,
)

######### Step 3 Count sleep, NREM, REM and wake
sleep_stages = ["NR", "N1", "R", "R1"]
nrem_stages = ["NR", "N1"]
rem_stages = ["R", "R1"]
wake_stages = ["W", "W1", "M"]

unstacked_stages = stage_df.unstack(level=0)
unstacked_stages.name = "stages"

names = {
    "total_name": "Total",
    "light_name": "Subjective_Light",
    "dark_name": "Subjective_Dark",
}

total_sleep = ((prep.lightdark_df(
    unstacked_stages,
    stage_list=sleep_stages,
    data_list=False,
    **names
)*4)/60)/60
nrem_sleep = ((prep.lightdark_df(
    unstacked_stages,
    stage_list=nrem_stages,
    data_list=False,
    **names
)*4)/60)/60
rem_sleep = ((prep.lightdark_df(
    unstacked_stages,
    stage_list=rem_stages,
    data_list=False,
    **names
)*4)/60)/60
wake_count = ((prep.lightdark_df(
    unstacked_stages,
    stage_list=wake_stages,
    data_list=False,
    **names
)*4)/60)/60

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

# stat constants
save_test_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                             "01_projects/01_thesisdata/03_lleeg/"
                             "03_analysis_outputs/05_figures/00_csvs/02_fig2")
stat_colnames = ["Day", "Animal", "Time", "Value"]
dep_var = stat_colnames[-1]
day = stat_colnames[0]
anim = stat_colnames[1]
time = stat_colnames[2]
time_vals = total_sleep.columns

# Repeated measures ANOVA for each of total/dark/light
ph_total_dict = {}
for part in time_vals:
    print(part)
    part_dir = save_test_dir / part
    # perform rm anova for each stage type
    ph_part_dict = {}
    for key, df in zip(totals_dict.keys(), totals_dict.values()):
        print(key)
        
        # tidy data
        long_df = df.stack().reset_index()
        long_df.columns = stat_colnames
        part_df = long_df.query("%s == '%s'" % (time, part))
        
        # do anova
        part_rm = pg.rm_anova(
            dv=dep_var,
            within=day,
            subject=anim,
            data=part_df
        )
        pg.print_table(part_rm)
        
        # do posthoc
        ph = pg.pairwise_tukey(
            dv=dep_var,
            between=day,
            data=part_df
        )
        pg.print_table(ph)
        ph_part_dict[key] = ph
        
        stage_test_dir = part_dir / key
        anova_file = stage_test_dir / "01_anova.csv"
        ph_file = stage_test_dir / "02_posthoc.csv"
        
        part_rm.to_csv(anova_file)
        ph.to_csv(ph_file)
    
    ph_part_df = pd.concat(ph_part_dict)
    ph_total_dict[part] = ph_part_df
ph_total_df = pd.concat(ph_total_dict)
ph_total_df = ph_total_df.reorder_levels([1, 0, 2])

################################################################################
# Plotting #
fig = plt.figure()

# Step 4 Plot Delta hypnogram left column
hypnogram_grid = gs.GridSpec(
    nrows=3,
    ncols=1,
    figure=fig,
    right=0.45
)

# plot each day of the hypnogram df of LL7 on separate axes
# create the axes and grab into a list
day_axes = [plt.subplot(x) for x in hypnogram_grid]

days = spectrum_df.index.get_level_values(1).unique()
ders = spectrum_df.index.get_level_values(2).unique()
label_col = -1
max_delta = delta_df.max()[0]
swe_ylabel = "Slow Wave Activity, µV$^2$/0.25Hz"
hypnogram_panels = ["A", "B", "C"]
hyp_pan_dict = dict(zip(days, hypnogram_panels))
panel_label_size = 10
panel_xpos = -0.1
panel_ypos = 1.1
leg_label_size = 8
markerscale = 0.5

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
        curr_ax.plot(
            curr_data,
            label=label
        )
    
    dark_index = curr_data.between_time("12:00:00", "23:59:00").index
    if day == days[0]:
        alpha = 0.3
        curr_ax.fill_between(
            dark_index,
            max_delta,
            0,
            facecolors='k',
            alpha=alpha
        )
    else:
        curr_ax.axvline(
            '2018-01-01 12:00:00',
            linestyle='--',
            color='k'
        )
    
    min_time = '2018-01-01 00:00:00'
    max_time = '2018-01-02 00:00:00'
    max_ylim = 4000
    curr_ax.set(
        xlim=[min_time, max_time],
        ylim=[0, max_ylim],
        title=day,
    )
    xfmt = mdates.DateFormatter("%H:%M")
    curr_ax.xaxis.set_major_formatter(xfmt)
    curr_ax.text(
        panel_xpos,
        panel_ypos,
        hyp_pan_dict[day],
        transform=curr_ax.transAxes,
        fontsize=panel_label_size
    )
    curr_ax.ticklabel_format(
        style='sci',
        axis='y',
        scilimits=(0, 0)
    )


day_axes[0].legend(
    loc=(2),
    ncol=3,
    fontsize=leg_label_size,
    markerscale=markerscale,
)
fig.autofmt_xdate()

day_axes[-1].set_xlabel(
    "Time of Day, ZT, hours",
    labelpad=10
)
day_axes[1].set_ylabel(
    swe_ylabel,
    labelpad=20
)

# Step 5 Plot totals right hand side - use seaborn
# create the axes to plot on
totals_grid = gs.GridSpec(
    nrows=4,
    ncols=1,
    figure=fig,
    left=0.55,
    hspace=0.15
)
totals_axes = [plt.subplot(x) for x in totals_grid]

# totals plotting constants
plotting_columns = ["Day", "Animal", "Light", "Sum"]
day_col = plotting_columns[0]
anim = plotting_columns[1]
lights = plotting_columns[2]
sum_data = plotting_columns[3]
order = names.values()
sig_col = "p-tukey"
sig_val = 0.05
sig_line_ylevel_day1 = 0.9
sig_line_ylevel_day2 = 0.95
time_ylabel = "Time in stage, hours"
plus_vals = [0, 0.3]
sig_colours = ["C1", "C2"]
tot_panels = ["D", "E", "F", "G"]
tot_panel_dict = dict(zip(totals_dict.keys(), tot_panels))

# plot each of total/light/dark on axes for each day
for count_label, curr_total_ax in zip(totals_dict.keys(), totals_axes):
    
    count_data = totals_dict[count_label]
    # tidy up the data
    # change units to s
    converted_count = count_data #((count_data * 4) / 60) / 60
    totals_plotting_data = converted_count.stack().reset_index()
    totals_plotting_data.columns = plotting_columns
    
    # plot data as a boxplot with individual points
    sns.boxplot(
        hue=day_col,
        y=sum_data,
        x=lights,
        data=totals_plotting_data,
        ax=curr_total_ax,
        order=order,
        fliersize=0
    )
    
    sns.stripplot(
        hue=day_col,
        y=sum_data,
        x=lights,
        data=totals_plotting_data,
        ax=curr_total_ax,
        dodge=True,
        order=order
    )
    
    # modify the legend
    ax_leg = curr_total_ax.legend()
    ax_leg.remove()
    
    # separate the days by a line
    curr_total_ax.axvline(0.5, color="k", alpha=0.5)
    curr_total_ax.axvline(1.5, color="k", alpha=0.5)
    
    # set REM ymax
    ymax = 16
    if count_label == "REM":
        ymax = 3
    curr_total_ax.set(ylim=[0, ymax], ylabel="", xlabel="")
    
    
    # set the title for the column
    if curr_total_ax == totals_axes[0]:
        curr_total_ax.set_title("Time in stage per part of day")
    if curr_total_ax != totals_axes[-1]:
        curr_total_ax.set_xticklabels("")
    
    curr_total_ax.text(
        1.0,
        0.5,
        count_label,
        transform=curr_total_ax.transAxes,
        rotation=270
    )
    curr_total_ax.text(
        panel_xpos,
        panel_ypos,
        tot_panel_dict[count_label],
        transform=curr_total_ax.transAxes,
        fontsize=panel_label_size
    )

    # add in statistical significance

    # get y value to lookup
    ycoord_data_val_day1 = plot.sig_line_coord_get(
        curr_total_ax,
        sig_line_ylevel_day1
    )
    ycoord_data_val_day2 = plot.sig_line_coord_get(
        curr_total_ax,
        sig_line_ylevel_day2
    )
    ycoord_list = [
        ycoord_data_val_day1,
        ycoord_data_val_day2
    ]
    
    # get x value from tests
    ph_stage = ph_total_df.loc[count_label]
    sig_day1 = plot.sig_locs_get(ph_stage)
    sig_day2 = plot.sig_locs_get(ph_stage, index_level2val=1)

    # get xdict to look up values
    label_loc_dict = plot.get_xtick_dict(curr_total_ax)
    
    # plot the significance lines
    for no, sig_dict in enumerate([sig_day1, sig_day2]):
        curr_plus_val = plus_vals[no]
        curr_ycoord = ycoord_list[no]
        curr_colour = sig_colours[no]
        plot.draw_sighlines(
            yval=curr_ycoord,
            sig_list=sig_day1,
            label_loc_dict=label_loc_dict,
            minus_val=0.3,
            plus_val=curr_plus_val,
            curr_ax=curr_total_ax,
            color=curr_colour
        )
        
# put the legend for the entire right column
ax_handles, ax_labels = curr_total_ax.get_legend_handles_labels()
fig.legend(
    handles=ax_handles[:3],
    loc=(0.9, 0.9),
    fontsize=leg_label_size,
    markerscale=markerscale
)
fig.text(
    -0.15,
    -0.12,
    time_ylabel,
    transform=totals_axes[1].transAxes,
    rotation=90,
    va='center'
)
fig.text(
    0.5,
    -0.5,
    "Part of day",
    transform=totals_axes[-1].transAxes,
    ha='center'
)

# set the figure title
fig.suptitle(
    "Constant light increases sleep during the active phase"
)
fig.set_size_inches(11.69, 8.27)

plt.savefig(SAVEFIG, dpi=600)


plt.close('all')
