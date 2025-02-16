import pandas as pd
import json
import requests
from requests.exceptions import RequestException

# Load CSV while handling quoted newlines
df = pd.read_csv("../data/asfi-sustainability-dataset/lists_2019_8.csv", encoding="Windows-1252", keep_default_na=False, dtype=str)

# Extract the the repo from the github url
df['repo'] = df['pj_github_url'].str.replace("https://github.com/apache/", "", regex=False)

# Read the 404 repo list from the file
with open("../data/github-pull-requests/404-not-found.txt", "r", encoding="utf-8") as f:
    not_found_repos = set(line.strip() for line in f if line.strip())  # Remove empty lines

# Check if repo is in the 404 list
df['github_url_error_404'] = df['repo'].isin(not_found_repos)

# Convert status from a number to a string
status_mapping = {"0": "incubating", "1": "graduated", "2": "retired"}
df['status'] = df['status'].map(status_mapping).fillna(df['status'])

# Rename rows
df['github_url'] = df['pj_github_url']


# Desired output order: repo, status, then the rest
desired_order = ['repo', 'status', 'listid', 'start_date', 'end_date', 'github_url', 'github_url_error_404']
df = df[[col for col in desired_order if col in df.columns]]

# Convert to JSON
json_output = df.to_json(orient="records", indent=2)

# Print the JSON output
print(json_output)
