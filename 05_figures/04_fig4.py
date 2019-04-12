# Figure 4 script
# Comparing the spectrum in zt 12-18 in the first two days in LL

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
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/actiPy")
import actiPy.episodes as ep

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
df_list = [prep.read_file_to_df(x, index_col=INDEX_COLS) for x in file_names]
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
    prep.read_file_to_df(x, index_col=[0]) for x in stage_names
]
stage_dfnames = [x.name for x in stage_list]
stage_dict = dict(zip(stage_dfnames, stage_list))
stage_df = pd.concat(stage_dict)
stage_df = stage_df.loc[:, :"LL_day2"]


# spectrum_df.loc[
#     idx["LL1", "LL_day2", :, :], :
# ] = spectrum_df.loc[
#     idx["LL1", "LL_day2", :, :], :
# ] * 10

# Step 2 select just the right ZTs
der = 'fro'
zt_slice = spectrum_df.loc[
           idx["LL2":, :, der, "2018-01-01 12:00:00":"2018-01-01 16:00:00"],
           :
           ]



# Step 3 Select just the state of interest - wake and NREM
nrem_mask = zt_slice["Stage"] == "NR"
nrem_slice = zt_slice.where(nrem_mask).drop("Stage", axis=1)
nrem_mean = nrem_slice.groupby(level=[0, 1, 2]).mean()
long_nrem = nrem_mean.iloc[:, 3:].stack().reset_index()
cols = ["Animal", "Light_period", "Derivation", "Frequency", "Power"]
long_nrem.columns = cols

# Step 4 Calculate episodes of activity per hour

# Only code NR episodes (NR and NR1)
nr_stages = ["NR", "N1"]
nr_int_df = stage_df.isin(nr_stages).astype(int)

# Use actiPy episode finder to get lengths and times of episodes
nr_episodes_df = nr_int_df.groupby(
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
    level=1
).sum()
long_sum = nr_ep_sum.stack().reset_index()

# get the mean duration using mean
nr_ep_mean = nr_episodes_df.groupby(
    level=0
).resample(
    "H",
    level=1
).mean()
long_mean = nr_ep_mean.stack().reset_index()
long_frag = long_sum.copy()
frag_cols = [
    "Animal",
    "Time",
    "Day",
    "Count"
]
long_frag.columns = frag_cols
long_frag["Mean"] = long_mean.iloc[:, -1]

# Step 5 Calculate delta SWA for NREM and wake

# Calculate SWA / hour
# Get sum and mean of artefact free delta power of a single derivation
band = ["Delta"]
range_to_sum = ("0.50Hz", "4.00Hz")
der = "fro"
swa_df = prep.create_df_for_single_band(
    spectrum_df,
    name_of_band=band,
    range_to_sum=range_to_sum,
    sum=False,
    mean=True
)
nrem_mask = swa_df["Stage"].isin(["NR"])
swa_nrem = swa_df.where(          # select only artefact free NREM
    nrem_mask,
    other=np.nan
).loc[
    idx[:, :, der, :],
    band
]

swa_nrem.loc[idx["LL1", "LL_day2", :, :"2018-01-01 23:59:59"], :
    ] = swa_nrem.loc[idx["LL1", "LL_day1", :, :], :] # Hack LL1 values for now

swa_hr = swa_nrem.groupby(        # Resample to hourly
    level=[0, 1]
).resample(
    "H",
    level=3,
    loffset=OFFSET,
).mean()

def norm_base_mean(protocol_df, baseline_str: str = "Baseline_0"):
    base_values = protocol_df.loc[idx[:, baseline_str], :]
    normalise_value = base_values.mean()
    normalised_df = (protocol_df / normalise_value) * 100
    return normalised_df

swa_hr_norm = swa_hr.groupby(
    level=[0]
).apply(
    norm_base_mean
)

swa_hr_shift = swa_hr_norm.shift(-1)
swa_hr_delta = ((swa_hr_shift - swa_hr_norm))# / swa_hr_norm) * 100
swa_hr_delta_mean = swa_hr_delta.groupby(level=[1, 2]).mean()
swa_hr_delta_long = swa_hr_delta_mean.reset_index()

# Calculate NREM / hour
nrem_stages = ["NR", "N1"]
nrem_int_df = stage_df.isin(nrem_stages).astype(int)
nr_hourly_perc = nrem_int_df.groupby(
    level=0
).resample(
    "H",
    level=1,
    loffset=OFFSET
).mean() * 100
nr_hourly_perc_mean = nr_hourly_perc.groupby(level=[1]).mean()
nr_hourly_perc_long = nr_hourly_perc_mean.stack(
).reorder_levels(
    [1, 0]
).sort_index(
).reset_index()

# Calculate Wake / hour
wake_stages = ["W", "W1"]
wake_int_df = stage_df.isin(wake_stages).astype(int)
wake_hourly_perc = wake_int_df.groupby(
    level=0
).resample(
    "H",
    level=1,
    loffset=OFFSET
).mean() * 100
wake_hourly_perc_mean = wake_hourly_perc.groupby(level=[1]).mean()
wake_hourly_perc_long = wake_hourly_perc_mean.stack(
).reorder_levels(
    [1, 0]
).sort_index(
).reset_index()

# Regression NREM ~ delta swa
swa_diss_df = swa_hr_delta_long.copy()
swa_diss_df["NR"] = nr_hourly_perc_long[nr_hourly_perc_long.columns[-1]]

# Regression wake ~ delta swa
swa_accum_df = swa_hr_delta_long.copy()
swa_accum_df["Wake"] = wake_hourly_perc_long[wake_hourly_perc_long.columns[-1]]

# Stats ########################################################################

# log transformed
# log_wake = np.log10(wake_spectrum)
# log_nrem = np.log10(nrem_spectrum)
#
# # 1. Does the day affect the spectrum?
# # repeated measures 2 way anova of power ~ freq x light period
#
# ders = log_wake.index.get_level_values(2).unique()
# freqs = log_wake.columns
# stats_colnames = ["Animal", "Time", "Derivation", "Frequency", "Power"]
# dep_var = stats_colnames[-1]
# day_col = stats_colnames[1]
# freq_col = stats_colnames[3]
# anim_col = stats_colnames[0]
# der_col = stats_colnames[2]
# data_list = [log_nrem, log_wake]
# data_type = ["NREM", "Wake"]
#
# save_test_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
#                              "01_projects/01_thesisdata/03_lleeg/"
#                              "03_analysis_outputs/05_figures/00_csvs/04_fig4")
#
# for df_type, label in zip(data_list, data_type):
#     print(label)
#     test_df = df_type.stack().reset_index()
#     test_df.columns = stats_colnames
#     type_dir = save_test_dir / label
#     for der in ders:
#         print(der)
#         der_dir = type_dir / der
#         der_df = test_df.query("%s == '%s'"%(der_col, der))
#         der_df = der_df.drop(der_col, axis=1)
#
#         # rm anova
#         anova = pg.rm_anova2(dv=dep_var,
#                              within=[day_col, freq_col],
#                              subject=anim_col,
#                              data=der_df)
#         pg.print_table(anova)
#         anova_file = der_dir / "01_anova.csv"
#         anova.to_csv(anova_file)
#
#         # post hoc
#         ph_dict = {}
#         for freq in freqs:
#             print(freq)
#             freq_df = der_df.query("%s == '%s'"%(freq_col, freq))
#             ph = pg.pairwise_tukey(dv=dep_var,
#                                    between=day_col,
#                                    data=freq_df)
#             pg.print_table(ph)
#             ph_dict[freq] = ph
#         ph_df = pd.concat(ph_dict)
#         ph_file = der_dir / "02_posthoc.csv"
#         ph_df.to_csv(ph_file)


# Step 6 Plot wake on top two and nrem on bottom two subplots

# tidy data further - remove _L1 and baselinse period
# baseline_wake_mask = long_wake["Light_period"] == "Baseline_-0"
# long_wake = long_wake.mask(baseline_wake_mask)
# ll1_wake_mask = long_wake["Animal"] == "LL1"
# long_wake = long_wake.mask(ll1_wake_mask)

# baseline_nrem_mask = long_nrem["Light_period"] == "Baseline_-0"
# long_nrem = long_nrem.mask(baseline_nrem_mask)
# ll1_nrem_mask = long_nrem['Animal'] == "LL1"
# long_nrem = long_nrem.mask(ll1_nrem_mask)


# figure constants
anim_col = cols[0]
animals = long_nrem[anim_col].unique()
lights = cols[1]
ders = cols[2]
freq = cols[3]
power = cols[4]
derivations = long_nrem[ders].unique()[1:]
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
xticks = np.linspace(0, 80, 21)
xticklabels = long_nrem[freq].unique()[2::4]
ylabel = "% change from Baseline day"
hline_kwargs = {"linestyle": "--",
                "color": "k"}

# Initialise figure
fig = plt.figure()

# plot the spectrum on the upper left
spectrum_grid = gs.GridSpec(
    ncols=1,
    nrows=1,
    figure=fig,
    right=0.45,
    bottom=0.55
)
spectrum_axes = [plt.subplot(x) for x in spectrum_grid]

curr_ax_spec = spectrum_axes[0]
sns.pointplot(
    x=freq,
    y=power,
    hue=lights,
    data=long_nrem,
    ax=curr_ax_spec,
    capsize=capsize,
    ci=sem,
    errwidth=errwidth
)
curr_ax_spec.set_yscale("log")


frag_grid = gs.GridSpec(
    nrows=2,
    ncols=1,
    figure=fig,
    hspace=0,
    left=0.55,
    bottom=0.55
)
frag_axes = [plt.subplot(x) for x in frag_grid]

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
 
# reg constants
swa_delta_col = swa_diss_df.columns[-2]
nr_col = swa_diss_df.columns[-1]
wake_col = swa_accum_df.columns[-1]
anim_col = swa_diss_df.columns[0]
day_col = swa_diss_df.columns[1]
time_col = swa_diss_df.columns[2]

reg_grid = gs.GridSpec(
    nrows=1,
    ncols=2,
    figure=fig,
    top=0.45
)
reg_axes = [plt.subplot(x) for x in reg_grid]

curr_ax_reg = reg_axes[0]

sns.scatterplot(
    x=nr_col,
    y=swa_delta_col,
    hue=anim_col,
    data=swa_diss_df,
    ax=curr_ax_reg
)
curr_ax_reg.legend().remove()

sns.scatterplot(
    x=wake_col,
    y=swa_delta_col,
    hue=anim_col,
    data=swa_accum_df,
    ax=reg_axes[1]
)


fig.set_size_inches(11.69, 8.27)

plt.close('all')
