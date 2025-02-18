# compute the time from pr creatation to first respond of the projects

import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import csv

# Load data from JSON files
with open('pull-requests.json', 'r') as pr_file:
    pull_requests = json.load(pr_file)

with open('comments.json', 'r') as comment_file:
    comments = json.load(comment_file)

with open('project-status.json', 'r') as status_file:
    project_status = json.load(status_file)

# Normalize repo names for case-insensitive matching
repo_status_dict = {entry["repo"].lower(): entry["status"] for entry in project_status}

# Create a dictionary to map issue_url to the earliest comment time
comment_times = {}
for comment in comments:
    issue_url = comment['issue_url']
    created_at = datetime.fromisoformat(comment['created_at'].replace('Z', '+00:00'))
    if issue_url not in comment_times or created_at < comment_times[issue_url]:
        comment_times[issue_url] = created_at

# Calculate response times per project
project_response_times = {}
for pr in pull_requests:
    repo_name = pr['repo'].lower()
    issue_url = pr['issue_url']
    if issue_url in comment_times:
        pr_created_at = datetime.fromisoformat(pr['created_at'].replace('Z', '+00:00'))
        first_comment_time = comment_times[issue_url]
        response_time = (first_comment_time - pr_created_at).total_seconds() / 3600  # Convert to hours
        
        if repo_name not in project_response_times:
            project_response_times[repo_name] = []
        project_response_times[repo_name].append(response_time)

# Compute average and median response time per project
project_avg_response_times = {repo: sum(times) / len(times) for repo, times in project_response_times.items()}
project_median_response_times = {repo: sorted(times)[len(times) // 2] for repo, times in project_response_times.items()}

graduated_projects = {}
retired_projects = {}
graduated_projects_median = {}
retired_projects_median = {}

for repo, avg_time in project_avg_response_times.items():
    status = repo_status_dict.get(repo, "unknown")
    if status == "graduated":
        graduated_projects[repo] = avg_time
        graduated_projects_median[repo] = project_median_response_times[repo]
    elif status == "retired":
        retired_projects[repo] = avg_time
        retired_projects_median[repo] = project_median_response_times[repo]

# Compute overall average and median response time for graduated and retired projects
def compute_average(times):
    return sum(times) / len(times) if times else None

def compute_median(times):
    sorted_times = sorted(times)
    n = len(sorted_times)
    if n == 0:
        return None
    return sorted_times[n // 2]

graduated_avg_response_time = compute_average(graduated_projects.values())
retired_avg_response_time = compute_average(retired_projects.values())

graduated_median_response_time = compute_median(graduated_projects_median.values())
retired_median_response_time = compute_median(retired_projects_median.values())

print("Average First Response Time per Project:")
for repo, avg_time in project_avg_response_times.items():
    print(f'{repo}: {avg_time:.2f} hours')

print("\nGraduated Projects - Average First Response Time per Project:")
for repo, avg_time in graduated_projects.items():
    print(f'{repo}: {avg_time:.2f} hours')

print("\nRetired Projects - Average First Response Time per Project:")
for repo, avg_time in retired_projects.items():
    print(f'{repo}: {avg_time:.2f} hours')

print(f'\nOverall Graduated Projects - Average First Response Time: {graduated_avg_response_time:.2f} hours' if graduated_avg_response_time else 'No data for graduated projects.')
print(f'Overall Retired Projects - Average First Response Time: {retired_avg_response_time:.2f} hours' if retired_avg_response_time else 'No data for retired projects.')

print(f'\nOverall Graduated Projects - Median First Response Time: {graduated_median_response_time:.2f} hours' if graduated_median_response_time else 'No data for graduated projects.')
print(f'Overall Retired Projects - Median First Response Time: {retired_median_response_time:.2f} hours' if retired_median_response_time else 'No data for retired projects.')

# Generate the bar graph for both average and median response times

categories = ["Graduated Projects", "Retired Projects"]
avg_latencies = [graduated_avg_response_time, retired_avg_response_time]
median_latencies = [graduated_median_response_time, retired_median_response_time]
x = np.arange(len(categories))  # label locations
width = 0.4  # width of the bars
plt.figure(figsize=(8, 5))
plt.bar(x - width/2, avg_latencies, width, label="Average", color='blue')
plt.bar(x + width/2, median_latencies, width, label="Median", color='orange')

plt.xlabel("Project Category")
plt.ylabel("First Response Time (hours)")
plt.title("Comparison of First Response Time by Project State")
plt.xticks(ticks=x, labels=categories)
plt.legend()
plt.savefig("first_respond_avg_median_plot.png", dpi=300, bbox_inches='tight')

# plt.show()

# Generate the box plot
data = [list(graduated_projects.values()), list(retired_projects.values())]
labels = ['Graduated', 'Retired']

plt.figure(figsize=(8, 5))
plt.boxplot(data, labels=labels)
plt.title("First Respond Time Distribution by Project State")
plt.ylabel("First Response Time (hours)")
plt.savefig("first_respond_box_plot.png", dpi=300, bbox_inches='tight')
# plt.show()

# Create KDE plot
plt.figure(figsize=(8, 5))

# Plot KDE for graduated projects
if len(graduated_projects) > 1:  # KDE requires at least two data points
    sns.kdeplot(list(graduated_projects.values()), label="Graduated", fill=True, color='blue')
# Plot KDE for retired projects
if len(retired_projects) > 1:
    sns.kdeplot(list(retired_projects.values()), label="Retired", fill=True, color='red')
# Add title and labels
plt.title("KDE Plot: First Respond Time by Project State")
plt.xlabel("Average First Respond Time (hours)")
plt.ylabel("Density")
plt.legend()
plt.savefig("first_respond_kde_plot.png", dpi=300, bbox_inches='tight')
# plt.show()

# Prepare data for CSV
csv_filename = "first_response_times.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Repository Name", "Status", "Number of PRs", "Average First Response Time (hours)", "Median First Response Time (hours)"])
    
    for repo, avg_time in project_avg_response_times.items():
        status = repo_status_dict.get(repo, "unknown")
        num_prs = len(project_response_times[repo])
        writer.writerow([repo, status, num_prs, f"{avg_time:.2f}", f"{project_median_response_times[repo]:.2f}"])


# # Generate the scatter plot
# graduated_x = ['Graduated'] * len(graduated_projects)
# graduated_y = list(graduated_projects.values())

# retired_x = ['Retired'] * len(retired_projects)
# retired_y = list(retired_projects.values())
# plt.figure(figsize=(8, 5))
# plt.scatter(graduated_x, graduated_y, color='blue', label="Graduated")
# plt.scatter(retired_x, retired_y, color='red', label="Retired")
# plt.title("Scatter Plot: Average Response Time (Graduated vs Retired)")
# plt.ylabel("Average Response Time (hours)")
# plt.legend()

# plt.show()