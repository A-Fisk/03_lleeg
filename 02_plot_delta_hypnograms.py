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
                         "/P3_LLEEG_Chapter3/01_data_files/07_clean_fft")
save_dir = input_dir.parents[1] / "03_analysis_outputs"
subdir_name = "01_delta_hypnograms"

read_kwargs = {
    "index_col":[0,1],
    "header":[0]
}

hypnogram_object = prep.SaveObjectPipeline(input_directory=input_dir,
                                      save_directory=save_dir,
                                      read_file_fx=(prep,
                                                    "read_file_to_df"),
                                      **read_kwargs)

plot_kwargs = {
    "data_list":None,
    "showfig":False,
    "savefig":True,
    "figsize":(10,10),
    "name_of_band":["Delta"],
    "range_to_sum":("0.50Hz", "4.00Hz"),
    "level_of_index":0,
    "label_col":-1,
    "base_freq":"4S",
    "plot_epochs":False,
    "set_file_title":True,
    "sharey":False,
    "legend":False
}

hypnogram_object.create_plot(function=(plot, "plot_hypnogram_from_df"),
                             subdir_name=subdir_name,
                             remove_col=False,
                             **plot_kwargs)
                            
