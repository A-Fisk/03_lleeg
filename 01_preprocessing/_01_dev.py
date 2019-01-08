# script for removing all headers and saving in new dir

import pathlib
import sys
import pandas as pd
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/07_python_package/"
                "sleepPy")
import sleepPy.preprocessing as prep

input_directory = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                               "01_projects/P3_LLEEG_Chapter3/01_data_files/"
                               "06_fft_files")
                              
save_directory = input_directory.parent
subdir_name = "07_clean_fft"

file_list = sorted(input_directory.glob("*.txt"))

kwargs = {
    "file":file_list[0],
    "header":17,
    "derivation_list":["fro", "occ", "foc"],
    "der_label":"Derivation",
    "time_index_column":(2),
    "test_index_range":[0,1,2,-2,-1]
}

final_df = prep.read_clean_fft_file(**kwargs)


