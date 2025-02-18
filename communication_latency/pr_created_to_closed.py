# compute the time from pr creatation to the pr close (resolution time)

import json
from datetime import datetime
from collections import defaultdict

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

# Function to compute PR resolution latency in hours, ignoring PRs with null "closed_at"
def compute_pr_resolution_latency(prs, repo_status_dict):
    graduated_latencies = []
    retired_latencies = []
    repo_latencies = defaultdict(list)

    for pr in prs:
        try:
            if pr.get("closed_at"):  # Only process closed PRs
                created_at = datetime.fromisoformat(pr["created_at"].replace("Z", "+00:00"))
                closed_at = datetime.fromisoformat(pr["closed_at"].replace("Z", "+00:00"))
                latency_hours = (closed_at - created_at).total_seconds() / 3600  # Convert seconds to hours

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
        if pr.get("closed_at"):
            if "repo" in pr:
                repo = pr["repo"]
                if repo not in repo_pr_stats:
                    repo_pr_stats[repo] = {"pr_count": 0, "avg_resolution": 0}
                repo_pr_stats[repo]["pr_count"] += 1

    for repo, latencies in repo_latencies.items():
        if repo in repo_pr_stats:
            repo_pr_stats[repo]["avg_resolution"] = sum(latencies) / len(latencies) if latencies else 0

    return repo_pr_stats

# Load repository status data from JSON
project_status_file_path = "project-status.json"
project_status_data = load_json_file(project_status_file_path)

# Create a dictionary mapping repo name to status
repo_status_dict = {entry["repo"]: entry["status"] for entry in project_status_data if "repo" in entry and "status" in entry}

# Load JSON data for PRs
json_file_path = "pull-requests.json"
pr_json_data = load_json_file(json_file_path)

# Compute latencies while ignoring PRs with null "closed_at"
graduated_latencies, retired_latencies, graduated_avg_latency, retired_avg_latency, repo_latencies = compute_pr_resolution_latency(pr_json_data, repo_status_dict)

# Compute PR stats (number of PRs and average resolution per repo)
repo_pr_stats = compute_pr_stats(pr_json_data, repo_latencies)

# # Print results
# print("PR Resolution Latency for Graduated Projects (in Hours, Ignoring PRs with null closed_at):")
# for pr_repo, pr_num, latency in graduated_latencies:
#     print(f"Repo: {pr_repo} PR #{pr_num}: {latency:.2f} hours")
# print("\n")

# print("PR Resolution Latency for Retired Projects (in Hours, Ignoring PRs with null closed_at):")
# for pr_repo, pr_num, latency in retired_latencies:
#     print(f"Repo: {pr_repo} PR #{pr_num}: {latency:.2f} hours")
# print("\n")

print(f"Average PR Resolution Latency for Graduated Projects: {graduated_avg_latency:.2f} hours")
print(f"Average PR Resolution Latency for Retired Projects: {retired_avg_latency:.2f} hours\n")

# Function to generate CSV file
with open("pr_resolution_time.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Repository Name", "Status", "Number of PRs", "Average Resolution Time (hours)"])
    
    for repo, stats in repo_pr_stats.items():
        status = repo_status_dict.get(repo, "unknown")
        writer.writerow([repo, status, stats["pr_count"], round(stats["avg_resolution"], 2)])



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

# Generate the bar graph

categories = ["Graduated Projects", "Retired Projects"]
avg_latencies = [graduated_avg_latency, retired_avg_latency]

plt.figure(figsize=(8, 5))
plt.bar(categories, avg_latencies, color=['blue', 'orange'])
plt.xlabel("Project Category")
plt.ylabel("Comparison of Average PR Resolution Time by Project State")
plt.title("Comparison of Average PR Resolution Time")
plt.savefig("pr_resolution_time_bar_plot.png", dpi=300, bbox_inches='tight')
# plt.show()


# Generate the box plot
graduated_resolutions = [stats["avg_resolution"] for stats in graduated_repos.values()]
retired_resolutions = [stats["avg_resolution"] for stats in retired_repos.values()]

plt.figure(figsize=(8, 6))

data = [graduated_resolutions, retired_resolutions]
labels = ["Graduated Projects", "Retired Projects"]
plt.boxplot(data, labels=labels, patch_artist=True)
plt.ylabel("PR Resolution Time Distribution by Project State")
plt.title("Box Plot of PR Resolution Times for Graduated vs Retired Projects")
plt.savefig("pr_resolution_time_boz_plot.png", dpi=300, bbox_inches='tight')
# plt.show()


# Create KDE plot

graduated_resolutions = [stats["avg_resolution"] for stats in graduated_repos.values() if stats["avg_resolution"] > 0]
retired_resolutions = [stats["avg_resolution"] for stats in retired_repos.values() if stats["avg_resolution"] > 0]

plt.figure(figsize=(8, 6))

# Plot KDE for graduated and retired projects
sns.kdeplot(graduated_resolutions, label="Graduated Projects", fill=True, alpha=0.5)
sns.kdeplot(retired_resolutions, label="Retired Projects", fill=True, alpha=0.5)

plt.xlabel("Average PR Resolution Time (Hours)")
plt.ylabel("Density")
plt.title("KDE Plot: PR Resolution Times by Project State")

plt.legend()
plt.savefig("pr_resolution_time_kde_plot.png", dpi=300, bbox_inches='tight')
# plt.show()