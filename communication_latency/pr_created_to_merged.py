import json
from datetime import datetime
from collections import defaultdict
import statistics

from matplotlib import pyplot as plt
import seaborn as sns
import csv

# Function to load JSON data from a file with error handling
def load_json_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return []

# Function to compute PR resolution latency in hours, ignoring PRs with null "merged_at"
def compute_pr_resolution_latency(prs, repo_status_dict):
    graduated_latencies = []
    retired_latencies = []
    repo_latencies = defaultdict(list)

    for pr in prs:
        try:
            if pr.get("merged_at"):  # Only process closed PRs
                created_at = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
                merged_at = datetime.fromisoformat(pr["merged_at"].replace("Z", "+00:00"))
                latency_hours = (merged_at - created_at).total_seconds() / 3600  # Convert seconds to hours

                status = repo_status_dict.get(pr.get("repo"), "unknown")
                repo_latencies[pr["repo"]].append(latency_hours)  # Store latencies for per-repo calculations

                if status == "graduated":
                    graduated_latencies.append((pr["repo"], pr["number"], latency_hours))
                elif status == "retired":
                    retired_latencies.append((pr["repo"], pr["number"], latency_hours))
        except (KeyError, ValueError) as e:
            print(f"Skipping PR due to missing/invalid data: {pr}. Error: {e}")

    # Compute average latency in hours
    graduated_avg_latency = (sum(lat[2] for lat in graduated_latencies) / len(graduated_latencies)) if graduated_latencies else 0
    retired_avg_latency = (sum(lat[2] for lat in retired_latencies) / len(retired_latencies)) if retired_latencies else 0

    return graduated_latencies, retired_latencies, graduated_avg_latency, retired_avg_latency, repo_latencies

# Function to compute the number of PRs per repository and average resolution time
def compute_pr_stats(prs, repo_latencies):
    repo_pr_stats = {}

    for pr in prs:
        if pr.get("merged_at"):
            if "repo" in pr:
                repo = pr["repo"]
                if repo not in repo_pr_stats:
                    repo_pr_stats[repo] = {"pr_count": 0, "avg_resolution": 0, "median_resolution": 0}
                repo_pr_stats[repo]["pr_count"] += 1

    for repo, latencies in repo_latencies.items():
        if repo in repo_pr_stats:
            repo_pr_stats[repo]["avg_resolution"] = sum(latencies) / len(latencies) if latencies else 0
            repo_pr_stats[repo]["median_resolution"] = statistics.median(latencies) if latencies else 0

    return repo_pr_stats

# Load repository status data from JSON
project_status_file_path = "project-status.json"
project_status_data = load_json_file(project_status_file_path)

# Create a dictionary mapping repo name to status
repo_status_dict = {entry["repo"]: entry["status"] for entry in project_status_data if "repo" in entry and "status" in entry}

# Load JSON data for PRs
json_file_path = "pull-requests.json"
pr_json_data = load_json_file(json_file_path)

# Compute latencies while ignoring PRs with null "cmerged_at"
graduated_latencies, retired_latencies, graduated_avg_latency, retired_avg_latency, repo_latencies = compute_pr_resolution_latency(pr_json_data, repo_status_dict)

# Compute PR stats (number of PRs and average resolution per repo)
repo_pr_stats = compute_pr_stats(pr_json_data, repo_latencies)

# # Print results
# print("PR Resolution Latency for Graduated Projects (in Hours, Ignoring PRs with null merged_at):")
# for pr_repo, pr_num, latency in graduated_latencies:
#     print(f"Repo: {pr_repo} PR #{pr_num}: {latency:.2f} hours")
# print("\n")

# print("PR Resolution Latency for Retired Projects (in Hours, Ignoring PRs with null merged_at):")
# for pr_repo, pr_num, latency in retired_latencies:
#     print(f"Repo: {pr_repo} PR #{pr_num}: {latency:.2f} hours")
# print("\n")

print(f"Average PR Merged Latency for Graduated Projects: {graduated_avg_latency:.2f} hours")
print(f"Average PR Merged Latency for Retired Projects: {retired_avg_latency:.2f} hours\n")

# Function to generate CSV file
with open("pr_merge_time.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Repository Name", "Status", "Number of PRs", "Average Time to Merge (hours)", "Median Merge Time (hours)"])
    
    for repo, stats in repo_pr_stats.items():
        status = repo_status_dict.get(repo, "unknown")
        writer.writerow([repo, status, stats["pr_count"], round(stats["avg_resolution"], 2), round(stats["median_resolution"], 2)])



# Separate repositories into graduated and retired projects
graduated_repos = {repo: stats for repo, stats in repo_pr_stats.items() if repo_status_dict.get(repo) == "graduated"}
retired_repos = {repo: stats for repo, stats in repo_pr_stats.items() if repo_status_dict.get(repo) == "retired"}

# Print Graduated Project PR Stats
print("\nGraduated Projects - PR Stats:")
for repo, stats in graduated_repos.items():
    print(f"Repo: {repo} | PR Count: {stats['pr_count']} | Avg Resolution: {stats['avg_resolution']:.2f} hours")

# Print Retired Project PR Stats
print("\nRetired Projects - PR Stats:")
for repo, stats in retired_repos.items():
    print(f"Repo: {repo} | PR Count: {stats['pr_count']} | Avg Resolution: {stats['avg_resolution']:.2f} hours")

# Compute median resolution times
def compute_median(latencies):
    if not latencies:
        return 0
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    mid = n // 2
    return (sorted_latencies[mid] if n % 2 != 0 else (sorted_latencies[mid - 1] + sorted_latencies[mid]) / 2)

graduated_median_latency = compute_median([lat[2] for lat in graduated_latencies])
retired_median_latency = compute_median([lat[2] for lat in retired_latencies])

print(f"Median PR Merged Latency for Graduated Projects: {graduated_median_latency:.2f} hours")
print(f"Median PR Merged Latency for Retired Projects: {retired_median_latency:.2f} hours\n")