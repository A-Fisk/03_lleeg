# script to run to check remove header working as expected
import pathlib
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/sleepPy")
import sleepPy.preprocessing as prep
import sleepPy.plots as plot

# define import dir
input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects"
                         "/P3_LLEEG_Chapter3/01_data_files/07_clean_fft")
save_dir = input_dir.parents[1] / "03_analysis_outputs"
subdir_name = "01_delta_hypnograms"

test_object = prep.SaveObjectPipeline(input_directory=input_dir,
                                      save_directory=save_dir,
                                      read_file_fx=(prep,
                                                    "read_file_to_df"),
                                      index_col=[2],
                                      header=[1])

test_object.create_plot(function_name="plot_spectral_hypnogram",
                        module=plot,
                        subdir_name=subdir_name,
                        data_list=test_object.df_list,
                        remove_col=False,
                        spectrum_name=["delta"],
                        spectrum_range=("0.50Hz", "4.00Hz"),
                        savefig=True,
                        figsize=(20,10))
                        
