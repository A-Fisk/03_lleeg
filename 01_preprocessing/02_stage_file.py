# script to run to check remove header working as expected
import pathlib
import sys
import pandas as pd
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/"
                   "07_python_package/sleepPy")
import sleepPy.preprocessing as prep

# define dirs
input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/01_projects"
                         "/01_thesisdata/03_lleeg/01_data_files"
                         "/07_clean_fft_files")
subdir_name = "08_stage_csv"
save_dir = input_dir.parent / subdir_name

# read in files
file_list = sorted(input_dir.glob("*.csv"))
file_names = [x.name for x in file_list]
index_col = [0, 1, 2]
df_list = [pd.read_csv(x, index_col=index_col) for x in file_list]

# grab the bits from each df
der = 'fro'
stage_df = [x.iloc[:, 0].unstack(level=0).loc[der] for x in df_list]

# save files
save_files = [(save_dir / x) for x in file_names]
stage_dict = dict(zip(save_files, stage_df))
for key, df in zip(stage_dict.keys(), stage_dict.values()):
    df.to_csv(key)
