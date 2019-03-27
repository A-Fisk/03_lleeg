# Script for creating figure 3
# Left hand column, hourly timecourse of sleep compared one day to the next
# right hand column cumulative sleep (hourly resampled), and cumulative delta
# power

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
                       "03_fig3.png")

# Step 1 Import files and tidy #################################################
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

# Step 2 Get hourly sleep data together ########################################
sleep_stages = ["NR", "N1", "R", "R1"]
sleep_int_df = stage_df.isin(sleep_stages).astype(int)
hourly_sleep_prop = sleep_int_df.groupby(level=0).resample("H", level=1).mean()

# put in with hourly mean and sem
hourly_mean = hourly_sleep_prop.stack().groupby(level=[2, 1]).mean()
hourly_sem = hourly_sleep_prop.stack().groupby(level=[2, 1]).sem()

hourly_sleep_df = pd.concat([hourly_mean, hourly_sem], axis=1)
hourly_columns = ["Mean", "SEM"]
hourly_sleep_df.columns = hourly_columns

# Step 3 Get cumulative NREM and delta power together ##########################
nrem_stages = ["NR", "N1"]
nrem_df = stage_df.isin(nrem_stages)
nrem_cumulative = nrem_df.groupby(level=0).cumsum()

# do cumulative hourly with mean and sem too
nrem_c_hourly = nrem_cumulative.groupby(level=0).resample("H", level=1).mean()
nrem_means = nrem_c_hourly.stack().groupby(level=[2, 1]).mean()
nrem_sems = nrem_c_hourly.stack().groupby(level=[2, 1]).sem()

nrem_mean_df = pd.concat([nrem_means, nrem_sems], axis=1)
nrem_mean_df.columns = hourly_columns

# create cumulative mean and hourly of delta power in NREM #####################
band = ["Delta"] # sum delta power
range_to_sum = ("0.50Hz", "4.00Hz")
delta_df = prep.create_df_for_single_band(spectrum_df, name_of_band=band,
                                         range_to_sum=range_to_sum)
swe_df = prep.create_df_for_single_band(spectrum_df,
                                        name_of_band=band,
                                        range_to_sum=range_to_sum,
                                        sum=False,
                                        mean=True)
nrem_mask = delta_df["Stage"].isin(nrem_stages[:-1])
nrem_delta = delta_df.where(nrem_mask, other=0)
nrem_swe = swe_df.where(nrem_mask, other=0)

# hack LL1 values wrong for now
nrem_delta.loc[idx["LL1", "LL_day2"]
    ] = nrem_delta.loc[idx["LL1", "LL_day1"]]
nrem_swe.loc[idx["LL1", "LL_day2"]
    ] = nrem_swe.loc[idx["LL1", "LL_day1"]]

# select single derivation and column
der = "fro"
nrem_delta_der = nrem_delta.loc[idx[:, :, der, :], band]
nrem_swe_der = nrem_swe.loc[idx[:, :, der, :], band]

# get hourly cumsum values
nrem_delta_cumsum = nrem_delta_der.groupby(level=[0, 1, 2]).cumsum()
nrem_delta_hourly = nrem_delta_cumsum.groupby(level=[0, 1]).resample(
    "H", level=3).mean()

# normalise to max delta value at each animals baseline day
data = nrem_delta_hourly
days = data.index.get_level_values(1).unique()
animals = data.index.get_level_values(0).unique()
data_list = []
for animal in animals:
    baseline = data.loc[idx[animal, days[0], "2018-01-01 23:00:00"],
               :].values[0][0]
    day_list = []
    for day in days:
        exp_day = data.loc[idx[animal, day, :, :], :]
        normalised = exp_day.div(baseline)
        day_list.append(normalised)
    day_df = pd.concat(day_list)
    data_list.append(day_df)
normalised_data = pd.concat(data_list)

# get mean and sem for each hour in a df
nrem_delta_hourly_means = normalised_data.groupby(level=[1, 2]).mean()
nrem_delta_hourly_sems = normalised_data.groupby(level=[1, 2]).sem()
nrem_delta_mean_df = pd.concat([nrem_delta_hourly_means,
                                nrem_delta_hourly_sems], axis=1)
nrem_delta_mean_df.columns = hourly_columns



# get hourly mean and SEM of straight delta power in a df

nd_hr_mean_anims = nrem_swe_der.groupby(level=[0, 1]
                                        ).resample("H",
                                                   level=3
                                                   ).mean()
mask = nd_hr_mean_anims[band] == 0.0
nd_hr_mean_anims = nd_hr_mean_anims.mask(mask, other=np.nan)
# normalise to baseline
# grab the mean of the baseline day, divide all values by that
def norm_to_base(anim_df,
                 baseline_str: str="Baseline_-0"):
    base_values = anim_df.loc[idx[:, baseline_str, :], :]
    normalise_values = base_values.mean()
    normalised_df = (anim_df / normalise_values) * 100
    return normalised_df

nd_hr_me_an_norm = nd_hr_mean_anims.groupby(level=0
                                            ).apply(norm_to_base)

# replace 0s with Nans
nan_mask = nd_hr_me_an_norm == 0.0
nd_hr_me_an_norm = nd_hr_me_an_norm.mask(nan_mask, other=np.nan)
# check if more than ? amount of NREM sleep per time bin
# create mask from sum of bool of resampled stage df, if above cutoff then
# include
# remove if less than 5 minutes of sleep per hour
epochs_5min = pd.Timedelta("10M").seconds / 4
bool_hr_mask = nrem_swe_der.astype(bool
                                   ).groupby(level=[0, 1]
                                             ).resample("H", level=3
                                                        ).sum() > epochs_5min
nd_hr_me_an_5min = nd_hr_me_an_norm.where(bool_hr_mask, other=np.nan)

# tidy into a df
nd_hr_mean = nd_hr_me_an_5min.groupby(level=[1, 2]).mean()
nd_hr_sem = nd_hr_me_an_5min.groupby(level=[1, 2]).sem()
nd_hr_df = pd.concat([nd_hr_mean, nd_hr_sem], axis=1)
nd_hr_df.columns = hourly_columns



############# Stats ############################################################

# 1. Does constant light affect the proportion of each hour spent asleep?
# Two way anova of HourxDay on proportion asleep
# Ignoring underlying rhythmic function here to do ANOVA and get a number

stat_colnames = ["Animal", "Hour", "Day", "Value"]
dep_var = stat_colnames[-1]
anim = stat_colnames[0]
hour_col = stat_colnames[1]
day_col = stat_colnames[2]
hours = hourly_sleep_prop.index.get_level_values(-1).unique()
save_test_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                             "01_projects/01_thesisdata/03_lleeg/"
                             "03_analysis_outputs/05_figures/00_csvs/03_fig3")
anova_csv = "01_anova.csv"
ph_csv = "02_posthoc.csv"

test_df = hourly_sleep_prop
long_df = test_df.stack().reset_index()
long_df.columns = stat_colnames

hourly_test_dir = save_test_dir / "hour_prop"

# prop 2 way rm
test_rm = pg.rm_anova2(dv=dep_var,
                       within=[day_col, hour_col],
                       subject=anim,
                       data=long_df)
pg.print_table(test_rm)

# prop post hoc
ph_dict = {}
for hour in hours:
    print(hour)
    hour_df = long_df.query("%s == '%s'"%(hour_col, hour))
    ph = pg.pairwise_tukey(dv=dep_var,
                           between=day_col,
                           data=hour_df)
    pg.print_table(ph)
    ph_dict[hour] = ph
ph_df = pd.concat(ph_dict)

hr_anova_file = hourly_test_dir / anova_csv
hr_ps_file = hourly_test_dir / ph_csv
test_rm.to_csv(hr_anova_file)
ph_df.to_csv(hr_ps_file)

# can't do repeated measures on swe since missing values
swe_test = nd_hr_me_an_5min.reset_index()
swe_test = swe_test.iloc[:, [0, 2, 1, 3]].copy()
swe_test.columns = stat_colnames

swa_test_dir = save_test_dir / "SWA"

swe_anova_df = swe_test.drop(anim, axis=1)
swe_anova = pg.anova(dv=dep_var,
                     between=[day_col, hour_col],
                     data=swe_anova_df)
pg.print_table(swe_anova)

# swe post hoc
ph_dict = {}
for hour in hours[:-1]:
    print(hour)
    hour_df = swe_anova_df.query("%s == '%s'"%(hour_col, hour))
    ph = pg.pairwise_tukey(dv=dep_var,
                           between=day_col,
                           data=hour_df)
    pg.print_table(ph)
    ph_dict[hour] = ph
ph_df = pd.concat(ph_dict)

swa_anova_file = swa_test_dir / anova_csv
swa_ps_file = swa_test_dir / ph_csv
swe_anova.to_csv(swa_anova_file)
ph_df.to_csv(swa_ps_file)

    
# 2. Does constant light change the amount or intensity of sleep?
# One way anova on the max values of time of NREM sleep and Delta power
# Post hoc - either linear regression or tukeys post hoc for each hour?
max_nrem_time = nrem_c_hourly.groupby(level=0).max()
test_df = max_nrem_time.stack().reset_index()
test_df.columns = [stat_colnames[x] for x in (0, 2, 3)]
test_df.drop(anim, axis=1, inplace=True)

nrem_test_dir = save_test_dir / "NREM"

# max nrem time anova
nrem_time_anova = pg.anova(dv=dep_var,
                           between=day_col,
                           data=test_df)
pg.print_table(nrem_time_anova)

# post hoc test for each hour of cumulative nrem
test_df = nrem_c_hourly.stack().reset_index()
test_df.columns = stat_colnames

ph_dict = {}
for hour in hours:
    print(hour)
    curr_hour_df = test_df.query("%s == '%s'"%(hour_col, hour))
    ph = pg.pairwise_tukey(dv=dep_var,
                           between=day_col,
                           data=curr_hour_df)
    pg.print_table(ph)
    ph_dict[hour] = ph
ph_df = pd.concat(ph_dict)

nrem_anova_file = nrem_test_dir / anova_csv
nrem_ps_file = nrem_test_dir / ph_csv
nrem_time_anova.to_csv(nrem_anova_file)
ph_df.to_csv(nrem_ps_file)

# same thing for max delta power
max_swa = normalised_data.groupby(level=[0, 1]).max()
test_df = max_swa.reset_index()
test_df.columns = [stat_colnames[x] for x in (0, 2, 3)]
test_df.drop(anim, axis=1, inplace=True)

swe_test_dir = save_test_dir / "SWE"

swa_anova = pg.anova(dv=dep_var,
                     between=day_col,
                     data=test_df)
pg.print_table(swa_anova)

# now do post hoc test
test_df = normalised_data.reset_index()
test_df.columns = stat_colnames

ph_dict = {}
for hour in hours[:-1]:
    print(hour)
    curr_hour_df = test_df.query("%s == '%s'"%(day_col, hour))
    ph = pg.pairwise_tukey(dv=dep_var,
                           between=hour_col,
                           data=curr_hour_df)
    pg.print_table(ph)
    ph_dict[hour] = ph
ph_df = pd.concat(ph_dict)

swe_anova_file = swe_test_dir / anova_csv
swe_ps_file = swe_test_dir / ph_csv
swa_anova.to_csv(swe_anova_file)
ph_df.to_csv(swe_ps_file)

################################################################################
# Step 4 plot
fig = plt.figure()

xfmt = mdates.DateFormatter("%H:%M:%S")
capsize = 5

# Plot LHS sleep time course
left_col = gs.GridSpec(nrows=2, ncols=1, figure=fig,
                       right=0.45)

# plot the hourly sleep per day on that axis
hourly_sleep_axis = [plt.subplot(x) for x in left_col]

days = hourly_sleep_df.index.get_level_values(0).unique()
hr_df_list = [hourly_sleep_df, nd_hr_df]

for curr_df, curr_ax in zip(hr_df_list, hourly_sleep_axis):
    
    for day in days:
        # select just the data to plot
        curr_day = curr_df.loc[day]
        
        mean_data = curr_day["Mean"]
        sem_data = curr_day["SEM"]
        
        curr_ax.errorbar(mean_data.index, mean_data.values,
                         yerr=sem_data,
                         marker='o',
                         label=day,
                         capsize=capsize)

        # set the xlimits
        xmin = "2018-01-01 00:00:00"
        xmax = "2018-01-02 00:00:00"
        
        # set the xlabel
        xlabel = "Time of day, ZT hours"
        curr_ax.set(xlim=[xmin, xmax],
                    xlabel=xlabel)
        
        # set times to look good
        curr_ax.set_xticklabels(curr_ax.get_xticklabels(),
                                rotation=30,
                                ha='right')
        curr_ax.xaxis.set_major_formatter(xfmt)
        
    dark_index = curr_day.loc["2018-01-01 12:00:00":"2018-01-02 00:00:00"].index
    alpha=0.1
    curr_ax.axvline("2018-01-01 12:00:00", color='k')
    curr_ax.fill_between(dark_index, 500, 0,
                                   facecolors='k', alpha=alpha)

# tidy hourly prop
hr_ax = hourly_sleep_axis[0]
hr_ax.legend()
prop_ylim = [0, 1]
hr_ax.set_ylim(prop_ylim)
hourly_ylabel = "Proportion of sleep per hour"
hourly_title = "Proportion of sleep per hour in constant light"
hr_ax.set(ylabel=hourly_ylabel,
          title=hourly_title)

# tidy swe
swe_ax = hourly_sleep_axis[1]
delt_ylim = [0, 250]
swe_ax.set_ylim(delt_ylim)
swe_ylabel = "SWE, % of entire baseline day"
swe_title = "SWE per hour"
swe_ax.set(ylabel=swe_ylabel,
           title=swe_title)


# Plot RHS cumulative sleep and delta

# create subplots on RHS
right_col = gs.GridSpec(nrows=2, ncols=1, figure=fig,
                        left=0.55)

top_ax = plt.subplot(right_col[0])
bottom_ax = plt.subplot(right_col[1])

# top plot do NREM time
for day in days:
    # select the data
    nrem_day = nrem_mean_df.loc[day]
    
    nrem_day = ((nrem_day * 4) / 60) / 60
    
    mean_nrem = nrem_day["Mean"]
    sem_nrem = nrem_day["SEM"]
    
    # plot with error bars
    top_ax.errorbar(mean_nrem.index, mean_nrem.values,
                    yerr=sem_nrem, marker='o', label=day,
                    capsize=capsize)
    
    # set xlimits
    # set ylabel
    nrem_ylabel = "Cumulative hours of NREM sleep"
    # set the title
    nrem_title = "Cumuative NREM sleep in constant light"
    # set ylims
    top_ymin, top_ymax = 0, 12
    top_ax.set(xlim=[xmin, xmax],
               ylim=[top_ymin, top_ymax],
               ylabel=nrem_ylabel,
               title=nrem_title)
    
    # set the legend
    top_ax.legend()
    
    # pretty times
    top_ax.set_xticklabels(top_ax.get_xticklabels(),
                           visible=False)
    top_ax.xaxis.set_major_formatter(xfmt)

# colour in dark
top_ax.axvline("2018-01-01 12:00:00", color='k')
top_ax.fill_between(dark_index, 15, 0, facecolor='k', alpha=alpha)

# bottom plot do delta
for day in days:
    # select the data
    delta_day = nrem_delta_mean_df.loc[day]
    
    mean_delta = delta_day["Mean"]
    sem_delta = delta_day["SEM"]
    
    bottom_ax.errorbar(mean_delta.index, mean_delta.values,
                       yerr=sem_delta, marker='o', label=day,
                       capsize=capsize)
    
    # set the limits
    # set the ylabel
    delta_ylabel = "Cumulative NREM Delta, % baseline "
    # set the title
    delta_title = "Cumuative Delta power in constant light"
    # set ylims
    bottom_ymin = 0
    bottom_ymax = 1.2
    bottom_ax.set(xlim=[xmin, xmax],
                  xlabel=xlabel,
                  ylim=[bottom_ymin, bottom_ymax],
                  ylabel=delta_ylabel,
                  title=delta_title)
    
    bottom_ax.set_xticklabels(bottom_ax.get_xticklabels(),
                              rotation=30, ha='right')
    bottom_ax.xaxis.set_major_formatter(xfmt)

delta_maxdouble = nrem_delta_mean_df.max()[0] * 1.5
bottom_ax.axvline("2018-01-01 12:00:00", color='k')
bottom_ax.fill_between(dark_index, delta_maxdouble, 0,
                       facecolor='k', alpha=alpha)
fig.suptitle("Constant light increases sleep")

fig.set_size_inches(11.69, 8.27)

plt.savefig(SAVEFIG, dpi=600)

plt.close('all')