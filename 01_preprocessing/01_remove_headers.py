# script for removing all headers and saving in new dir

import pathlib
import sys
sys.path.insert(0, "/Users/angusfisk/Documents/01_PhD_files/07_python_package/"
                "sleepPy")
import sleepPy.preprocessing as prep

input_directory = pathlib.Path("/Users/angusfisk/Documents/01_PhD_files/"
                               "01_projects/P3_LLEEG_Chapter3/01_data_files/"
                               "06_fft_files")
                              
save_directory = input_directory.parent
subdir_name = "07_clean_fft"

clean_object = prep.SaveObjectPipeline(input_directory=input_directory,
                                       save_directory=save_directory,
                                       search_suffix=".txt",
                                       readfile=False)
save_dir_path = prep.create_subdir(save_directory,subdir_name)
kwargs = {"save_dir_path":save_dir_path,
          "save_suffix_file":"_clean.csv"}
clean_object.process_file(module=prep,
                          function_name="remove_header_and_save",
                          subdir_name=subdir_name,
                          savecsv=False,
                          object_list=clean_object.file_list,
                          **kwargs)



