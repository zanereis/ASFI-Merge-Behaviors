# ASFI-Merge-Behaviors
Companion to "Exploring Pull Requests and Project Sustainability In Apache Incubator Projects"

# Installation
1. Install git in your system if it is not already installed. Go to https://git-scm.com/downloads for installing git.
2. Then go to a directory where you want to download the project

## Clone repository
```
git clone https://github.com/zanereis/ASFI-Merge-Behaviors.git
```

## Go to the root directory of the project
```
cd ASFI-Merge-Behaviors
```

## Install all the required packages
```
pip install -r requirements.txt
```

### Note: Make sure you have the ASFI dataset and the data from issues of this repository downloaded and saved in your desired directory

### Analysis PR related metrics with pull request data
use the  pr_data_analysis directory
need pull_requests.json, project-status.json from https://github.com/zanereis/ASFI-Merge-Behaviors/releases/tag/0.0.1
need monthly_data.json from https://github.com/zanereis/ASFI-Merge-Behaviors/tree/main/data/monthly_data
need this project list csv from ASFI https://github.com/zanereis/ASFI-Merge-Behaviors/blob/main/data/asfi-sustainability-dataset/lists_2019_8.csv

Generating initial summary
```
python3 data_reader.py
```

Generating initial plots
```
python3 graph_grenerate.py
```

Generating normalized summary
```
python3 normalizer.py
```

Generating normalized summary
```
python3 normalized_graph_generate.py
```

### Analyzing communication latency
use thecommunication_latency directory
Put all the files from https://github.com/zanereis/ASFI-Merge-Behaviors/releases/tag/0.0.1 in this directory
After that run all the python files in the directory to generate graphs 

### Analysing the core developers with pull requests
All analyses regarding the core developers can be found in the "core_developers" folder.  

### Analysing the datasets (both the ASFI dataset and the ones made by us) 
Run core_developers/1.pre-check.ipynb
```
jupyter nbconvert --to notebook --execute core_developers/1.pre-check.ipynb
```

### Preprocessing the datasets and creating the project_core_dev.csv and pr_core_dev.csv
Run core_developers/2.preprocessing.ipynb
```
jupyter nbconvert --to notebook --execute core_developers/2.preprocessing.ipynb
```

### Exploratory data analysis on the core developers with pull requests
```
jupyter nbconvert --to notebook --execute core_developers/3.eda.ipynb
```

### For lasso regression,p_values,ML model and SHAP analysis go to model directory. Have to make sure the monthly_data.json is in the right directory.
### We just have to run the respective ml model preprocess_train_'model_name' files