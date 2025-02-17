# compute the time from pr creatation to first respond of the projects

import json
from datetime import datetime

from matplotlib import pyplot as plt
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

# Compute average response time per project
project_avg_response_times = {repo: sum(times) / len(times) for repo, times in project_response_times.items()}

graduated_projects = {}
retired_projects = {}
for repo, avg_time in project_avg_response_times.items():
    status = repo_status_dict.get(repo, "unknown")
    if status == "graduated":
        graduated_projects[repo] = avg_time
    elif status == "retired":
        retired_projects[repo] = avg_time

# Compute the overall average response time for graduated and retired projects
def compute_average(times):
    return sum(times) / len(times) if times else None

graduated_avg_response_time = compute_average(graduated_projects.values())
retired_avg_response_time = compute_average(retired_projects.values())

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

# Prepare data for CSV
csv_filename = "first_response_times.csv"
with open(csv_filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Repository Name", "Status", "Number of PRs", "Average First Response Time (hours)"])
    
    for repo, avg_time in project_avg_response_times.items():
        status = repo_status_dict.get(repo, "unknown")
        num_prs = len(project_response_times[repo])
        writer.writerow([repo, status, num_prs, avg_time])

# Generate the bar graph

categories = ["Graduated Projects", "Retired Projects"]
avg_latencies = [graduated_avg_response_time, retired_avg_response_time]

plt.figure(figsize=(8, 5))
plt.bar(categories, avg_latencies, color=['blue', 'orange'])
plt.xlabel("Project Category")
plt.ylabel("Average First Respond Time (hours)")
plt.title("Comparison of First Respond Time")

plt.show()

# Generate the box plot
data = [list(graduated_projects.values()), list(retired_projects.values())]
labels = ['Graduated', 'Retired']

plt.figure(figsize=(8, 5))
plt.boxplot(data, labels=labels)
plt.title("First Respond Time Distribution: Graduated vs Retired Projects")
plt.ylabel("First Response Time (hours)")

plt.show()

# Create KDE plot
plt.figure(figsize=(8, 5))

# Plot KDE for graduated projects
if len(graduated_projects) > 1:  # KDE requires at least two data points
    sns.kdeplot(list(graduated_projects.values()), label="Graduated", fill=True, color='blue')
# Plot KDE for retired projects
if len(retired_projects) > 1:
    sns.kdeplot(list(retired_projects.values()), label="Retired", fill=True, color='red')
# Add title and labels
plt.title("KDE Plot: First Respond Time (Graduated vs Retired)")
plt.xlabel("Average First Respond Time (hours)")
plt.ylabel("Density")
plt.legend()
plt.show()


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