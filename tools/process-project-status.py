import pandas as pd
import json
import requests
from requests.exceptions import RequestException

# Load CSVs
df = pd.read_csv("../data/asfi-sustainability-dataset/lists_2019_8.csv", encoding="Windows-1252", keep_default_na=False, dtype=str)
corrected_urls = pd.read_csv("../data/asfi-sustainability-dataset/correct-github-urls.csv", encoding="utf-8", keep_default_na=False, dtype=str) 

# Read the 404 repo list from the file
with open("../data/github-pull-requests/404-not-found.txt", "r", encoding="utf-8") as f:
    not_found_repos = set(line.strip() for line in f if line.strip())  # Remove empty lines

# Generate 'repo' column based on pj_github_url
df['repo'] = df['pj_github_url'].str.replace("https://github.com/apache/", "", regex=False)

# Check if repo is in the 404 list
df['github_url_error_404'] = df['repo'].isin(not_found_repos)

# Convert status from a number to a string
status_mapping = {"0": "incubating", "1": "graduated", "2": "retired"}
df['status'] = df['status'].map(status_mapping).fillna(df['status'])


## Correct github urls

# Convert to dictionary for quick lookup (listid → pj_github_url)
url_mapping = dict(zip(corrected_urls['listid'], corrected_urls['pj_github_url']))

# Update pj_github_url where listid matches
df['pj_github_url'] = df.apply(lambda row: url_mapping.get(row['listid'], row['pj_github_url']), axis=1)

# Set github_url_error_404 to False for corrected list IDs
df.loc[df['listid'].isin(corrected_urls['listid']), 'github_url_error_404'] = False

# Explicitely mark XMLBeans/C++ to not found
df.loc[df['repo'] == "XMLBeans/C++", 'github_url_error_404'] = True


# Regenerate 'repo' column based on updated pj_github_url
df['repo'] = df['pj_github_url'].str.replace("https://github.com/apache/", "", regex=False)

# Rename rows
df['github_url'] = df['pj_github_url']

# Desired output order: repo, status, then the rest
desired_order = ['repo', 'status', 'listid', 'listname', 'start_date', 'end_date', 'github_url', 'github_url_error_404']
df = df[[col for col in desired_order if col in df.columns]]

# Convert to JSON
json_output = df.to_json(orient="records", indent=2)

## Count the number of True and False values in 404_not_found
#num_404_true = df['github_url_error_404'].sum()  # Count where 404_not_found is True
#num_404_false = len(df) - num_404_true    # Count where 404_not_found is False
#total_entries = len(df)                   # Total number of entries
#
## Print the results
#print(f"Entries where 404_not_found = True: {num_404_true}")
#print(f"Entries where 404_not_found = False: {num_404_false}")
#print(f"Total entries: {total_entries}")

# Print the JSON output
print(json_output)
