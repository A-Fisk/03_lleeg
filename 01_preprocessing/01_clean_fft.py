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

kwargs = {
      "save_suffix_file":"_clean.csv",
      "savecsv":True,
      "object_list":clean_object.file_list,
      "header":17,
      "derivation_list":["fro", "occ", "foc"],
      "der_label":"Derivation",
      "time_index_column":(2),
      "test_index_range":[0,1,2,-2,-1]
  }

clean_object.process_file(module=prep,
                          function_name="read_clean_fft_file",
                          subdir_name=subdir_name,
                          **kwargs)



