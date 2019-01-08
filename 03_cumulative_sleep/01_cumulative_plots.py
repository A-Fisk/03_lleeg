# script to run to check remove header working as expected
import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/sleepPy")
import sleepPy.preprocessing as prep
import sleepPy.plots as plot
import seaborn as sns
sns.set()

# define import dir
input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects"
                         "/P3_LLEEG_Chapter3/01_data_files/07_clean_fft_files/")
save_dir = input_dir.parents[1]
subdir_name = "01_data_files/08_stage_csv"
plot_subdir_name = "03_analysis_outputs/02_cumulative_plots/01_cumulative_sleep"
plot_subdir_path = prep.create_subdir(save_dir, plot_subdir_name)

init_kwargs = {
    "input_directory":input_dir,
    "save_directory":save_dir,
    "subdir_name":subdir_name,
    "func":(prep, "read_file_to_df"),
    "search_suffix":".csv",
    "readfile":True,
    "index_col":[0,1,2],
    "header":[0]
}
cumulative_sleep_object = prep.SaveObjectPipeline(**init_kwargs)

process_kwargs ={
    "function": (prep, "create_stage_df"),
    "savecsv": True,
}
cumulative_sleep_object.process_file(**process_kwargs)

plot_kwargs = {
    "function": (plot, "plot_cumulative_from_stage_df"),
    "data_list": cumulative_sleep_object.processed_list,
    "remove_col": False,
    "subdir_path": plot_subdir_path,
    "stages": ["NR", "R", "NR1", "R1"],
    "base_freq": "4S",
    "target_freq": "1H",
    "showfig": False,
    "savefig": True,
    "legend_loc": "center right",
    "figsize": (10,10),
    "ylabel": "Cumulative sleep, (hours)"
}
cumulative_sleep_object.create_plot(**plot_kwargs)
