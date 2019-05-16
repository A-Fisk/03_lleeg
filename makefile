SHELL=/bin/bash

all: create actograms episodes summary waveform 

create:
	@echo "activating env"
	@echo "creating files"
	source activate sleepPy_environment; \
	cd /Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/03_lleeg/02_analysis_files/01_preprocessing; \
	ls *.py; \
	python 01_clean_fft.py; \
	python 02_stage_file.py; \

actograms:
	@echo "activating env"
	@echo "running actograms"; \
	source activate actiPy_environment; \
	cd /Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/02_circdis/02_analysis_files/02_activity/01_actograms; \
	ls *.py; \
	python 01_longactogram.py; \
	python 02_shortactogram.py; \


episodes:
	@echo "activating env"
	@echo "running episodes"; \
	source activate actiPy_environment; \
	cd /Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/02_circdis/02_analysis_files/02_activity/02_episodes; \
	ls *.py; \
	python 01_create_files.py; \
	python 02_histograms.py; \
	python 04_histogram_sum.py


summary:
	@echo "activating env"
	@echo "running summaries"; \
	source activate actiPy_environment; \
	cd /Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/02_circdis/02_analysis_files/02_activity/03_summary_stats; \
	python *.py

waveform:
	@echo "activating env"
	@echo "running waveform"; \
	source activate actiPy_environment; \
	cd /Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/02_circdis/02_analysis_files/02_activity/04_mean_waveform; \
	python *.py

figures:
	@echo "activating env"
	@echo "creating files"
	source activate sleepPy_environment; \
	cd /Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/03_lleeg/02_analysis_files/05_figures; \
	ls *.py; \
	python 01_fig1.py; \
	python 02_fig2.py; \
	python 03_fig3.py; \
	python 04_fig4.py; \



