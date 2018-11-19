# script for removing all headers and saving in new dir

import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/07_python_package/"
                "sleepPy")
import sleepPy.preprocessing as prep

input_dir = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                               "01_projects/P3_LLEEG_Chapter3/01_data_files/"
                               "06_fft_files")
                              
save_dir = input_dir.parent
subdir_name = "07_clean_fft_files"

clean_object = prep.SaveObjectPipeline(input_directory=input_dir,
                                      save_directory=save_dir,
                                      search_suffix=".txt",
                                      readfile=False,
                                      subdir_name=subdir_name)

animal_file_list = prep.create_dict_of_animal_lists(clean_object.file_list,
                                                    input_dir,
                                                    anim_range=(0,3))
                                                    
kwargs = {
    "save_suffix_file": "_clean.csv",
    "savecsv": True,
    "function": (prep, "single_df_for_animal"),
    "object_list": animal_file_list.values(),
    "file_list": animal_file_list.keys(),
    "header": 17,
    "derivation_list": ["fro", "occ", "foc"],
    "der_label": "Derivation",
    "time_index_column": (2),
    "test_index_range": [0,1,2,-2,-1],
    "period_label": "Light_period",
    "anim_range": (0,3),
    "day_range": (-6)
}
clean_object.process_file(**kwargs)
