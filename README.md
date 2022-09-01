# Lexical Stability
Code for the paper "Lexical Stability of Psychiatric Clinical Notes from Electronic Health Records over a Decade". The raw data can not be shared, but aggregated data (keyword counts, descriptive statistics, novelty) can be under `data`. `r_src/changepoint_analysis.Rmd` can be run to recreate the analysis from the paper. 


## Pipeline
1. `src/extract_descriptive_statistics.py` to extract descriptive statistics with `textdescriptives`.
2. `src/summarize_descriptive_statistics.py` to aggregate them in quarterly bins.
3. `src/get_keyword_counts.py` to extract keyword counts and aggregate them quarterly.
4. `src/extract_novelty.py` to calculate novelty based on the keyword proportions.
5. `r_src/changepoint_analysis.Rmd` to conduct the changepoint analysis.



## Directory structure
├── data/
│   ├── entropy_prop.csv # novelty
│   ├── keyword_counts.pkl # keyword counts
│   ├── keywords.yaml # the extracted keywords
│   └── td_stats.csv # extracted descriptive statistics
├── pretty_path.py
├── r_src/ # R scripts
│   ├── changepoint_analysis.Rmd # main analysis
│   ├── novelty_visualization.Rmd # to reproduce figure 5
│   ├── number_of_notes.Rmd # to reproduce table 2
│   └── r_utils/
│       └── change_point_detection.R # functions for change point detection
├── README.md
├── requirements.txt
├── src/ # python scripts
│   ├── __init__.py
│   ├── extract_descriptive_statistics.py 
│   ├── extract_novelty.py
│   ├── figures_and_tables/ # 
│   │   ├── dep_visualization.py # to reproduce figure 6
│   │   ├── note_description.py # to create data for table 2
│   │   └── patient_description.py # to reproduce table 1
│   ├── get_keyword_counts.py
│   ├── summarize_descriptive_statistics.py
│   └── utils/
│       ├── __init__.py
│       ├── infodynamics.py # code for calculating novelty
│       └── utils.py # misc. utilities
