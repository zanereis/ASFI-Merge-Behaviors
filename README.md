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

## Analysing the core developers with pull requests
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
