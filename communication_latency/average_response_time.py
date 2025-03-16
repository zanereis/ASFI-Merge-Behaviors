import json
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON data
with open("comments.json", "r") as f:
    comments = json.load(f)

with open("pull-requests.json", "r") as f:
    pull_requests = json.load(f)

with open("project-status.json", "r") as f:
    project_status = json.load(f)

# Create a dictionary to map repos to their status
repo_status = {proj["repo"]: proj["status"] for proj in project_status}

# Organize comments by repo
repo_comments = {}
for comment in comments:
    repo = comment["repo"]
    created_at = datetime.strptime(comment["created_at"], "%Y-%m-%dT%H:%M:%SZ")
    if repo not in repo_comments:
        repo_comments[repo] = []
    repo_comments[repo].append(created_at)

# Sort comments by timestamp within each repo
for repo in repo_comments:
    repo_comments[repo].sort()

# Compute time differences between consecutive comments in hours
repo_time_diffs = {}
for repo, timestamps in repo_comments.items():
    if len(timestamps) > 1:
        time_diffs = [(timestamps[i] - timestamps[i - 1]).total_seconds() / 3600 for i in range(1, len(timestamps))]
        repo_time_diffs[repo] = time_diffs

# Compute statistics for each repo
repo_stats = {}
for repo, diffs in repo_time_diffs.items():
    mean_diff = round(np.mean(diffs), 2)
    median_diff = round(np.median(diffs), 2)
    percentile_90 = round(np.percentile(diffs, 90), 2)
    status = repo_status.get(repo, "Unknown")
    repo_stats[repo] = {"status": status, "mean_hours": mean_diff, "median_hours": median_diff, "90th_percentile_hours": percentile_90}

# Collect mean and median times for graduated and retired projects
graduated_means = []
graduated_medians = []
retired_means = []
retired_medians = []

for repo, stats in repo_stats.items():
    if stats["status"] == "graduated":
        graduated_means.append(stats["mean_hours"])
        graduated_medians.append(stats["median_hours"])
    elif stats["status"] == "retired":
        retired_means.append(stats["mean_hours"])
        retired_medians.append(stats["median_hours"])

# Compute mean and median across all mean and median values
if graduated_means and graduated_medians:
    graduated_overall_mean = round(np.mean(graduated_means), 2)
    graduated_overall_median = round(np.median(graduated_medians), 2)
    print(f"Graduated Projects - Overall Mean of Mean Time Between Comments: {graduated_overall_mean} hours")
    print(f"Graduated Projects - Overall Median of Median Time Between Comments: {graduated_overall_median} hours")
else:
    print("No graduated projects with sufficient data available.")

if retired_means and retired_medians:
    retired_overall_mean = round(np.mean(retired_means), 2)
    retired_overall_median = round(np.median(retired_medians), 2)
    print(f"Retired Projects - Overall Mean of Mean Time Between Comments: {retired_overall_mean} hours")
    print(f"Retired Projects - Overall Median of Median Time Between Comments: {retired_overall_median} hours")
else:
    print("No retired projects with sufficient data available.")

# Print results for each repo
for repo, stats in repo_stats.items():
    print(f"Repo: {repo}")
    print(f"  Status: {stats['status']}")
    print(f"  Mean Time Between Comments: {stats['mean_hours']} hours")
    print(f"  Median Time Between Comments: {stats['median_hours']} hours")
    print(f"  90th Percentile Time Between Comments: {stats['90th_percentile_hours']} hours")
    print("-")

# Separate mean response times for graduated and retired projects
graduated_means = [stats["mean_hours"] for repo, stats in repo_stats.items() if stats["status"] == "graduated"]
retired_means = [stats["mean_hours"] for repo, stats in repo_stats.items() if stats["status"] == "retired"]

# Create box plot
plt.figure(figsize=(8, 6))
plt.boxplot([graduated_means, retired_means], labels=["Graduated", "Retired"])

# Set labels and title
plt.ylabel("Average Response Time (hours)")
plt.title("Comparison of Average Response Time Between Graduated and Retired Projects")

# Show the plot
plt.show()

# Save results to CSV
csv_filename = "average_comment_time.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Project Names", "Status", "Mean Time Between Comments(hours)", "Median Time Between Comments(hours)", "90th Percentile Time Between Comments(hours)"])
    for repo, stats in repo_stats.items():
        writer.writerow([repo, stats["status"], stats["mean_hours"], stats["median_hours"], stats["90th_percentile_hours"]])

print(f"Results saved to {csv_filename}")