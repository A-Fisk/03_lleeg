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
                         "/P3_LLEEG_Chapter3/01_data_files/07_clean_fft_files")
save_dir = input_dir.parents[1] / "03_analysis_outputs"
subdir_name = "01_delta_hypnograms"

init_kwargs = {
    "input_directory": input_dir,
    "save_directory": save_dir,
    "subdir_name": subdir_name,
    "func": (prep, "read_file_to_df"),
    "search_suffix": ".csv",
    "readfile": True,
    "index_col": [0, 1, 2],
    "header": [0]
}
hypnogram_object = prep.SaveObjectPipeline(**init_kwargs)

process_kwargs = {
    "function": (prep, "_sep_by_top_index"),
    "savecsv": False
}
hypnogram_object.process_file(**process_kwargs)

plot_kwargs = {
    "function": (plot, "plot_hypnogram_from_list"),
    "remove_col": False,
    "data_list": hypnogram_object.processed_list,
    "showfig": False,
    "savefig": True,
    "figsize": (10, 10),
    "name_of_band": ["Delta"],
    "range_to_sum": ("0.50Hz", "4.00Hz"),
    "level_of_index": 0,
    "label_col": -1,
    "base_freq": "4S",
    "plot_epochs": False,
    "set_file_title": True,
    "set_name_title": False,
    "sharey": False,
    "legend": True
}
# Now loop over every object in test_processed_list and plot_hypnogram
hypnogram_object.create_plot(**plot_kwargs)
