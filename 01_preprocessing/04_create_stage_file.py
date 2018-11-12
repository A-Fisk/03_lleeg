# script to run to check remove header working as expected
import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/sleepPy")
import sleepPy.preprocessing as prep

# define import dir
input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects"
                         "/P3_LLEEG_Chapter3/01_data_files/07_clean_fft"
                         "/01_reindexed")
save_dir = input_dir.parents[1]
subdir_name = "08_stage_df"

kwargs = {"index_col":0,
          "header":0,
          "check_cols":False,
          "rename_cols":False,
          "drop_cols":False}

prep.create_stage_csv(input_dir=input_dir,
                      save_dir=save_dir,
                      subdir_name=subdir_name,
                      **kwargs)



