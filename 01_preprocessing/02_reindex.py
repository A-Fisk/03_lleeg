# Script to reset the index so always goes 00:00:00 - 00:00:00
# shows better reflection of circadian/zeitgeber time
import pathlib
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/sleepPy")
import sleepPy.preprocessing as prep

# define import dir
input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects"
                         "/P3_LLEEG_Chapter3/01_data_files/07_clean_fft")
save_dir = input_dir
subdir_name = "01_reindexed"

reindex_object = prep.SaveObjectPipeline(input_directory=input_dir,
                                         save_directory=save_dir,
                                         read_file_fx=(prep,
                                                       "read_file_to_df"),
                                         index_col=[2],
                                         header=[1])

reindex_object.process_file(prep,
                            "reindex_file",
                            subdir_name=subdir_name,
                            save_suffix=".csv",
                            savecsv=True)

