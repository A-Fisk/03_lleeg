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
                         "/P3_LLEEG_Chapter3/01_data_files/08_stage_df")
save_dir = input_dir.parents[1] / "03_analysis_outputs/02_cumulative_plots"
subdir_name = "01_cumulative_sleep"

kwargs = {"index_col":0,
          "header":0,
          "check_cols":False,
          "rename_cols":False,
          "drop_cols":False}

plot_object = prep.SaveObjectPipeline(input_directory=input_dir,
                                      save_directory=save_dir,
                                      read_file_fx=(prep,
                                                    "read_file_to_df"),
                                      **kwargs)

plot_kwargs = {"savefig":True,
               "figsize":(10,10),
               "stages":["NR","R","NR1","R1"],
               "base_freq":"4S",
               "target_freq":"1H",
               "ylabel":"Cumulative sleep (hours)"}

plot_object.create_plot(function_name="plot_cumulative_from_stage_df",
                        module=plot,
                        subdir_name=subdir_name,
                        data_list=plot_object.df_list,
                        remove_col=False,
                        **plot_kwargs)