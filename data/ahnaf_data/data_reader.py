import json
import ijson
import csv
from collections import defaultdict

def read_json_status(json_filename):
    status_data = {}
    status_mapping = {"incubating": 0, "graduated": 1, "retired": 2}
    try:
        with open(json_filename, 'r', encoding='utf-8') as jsonfile:
            projects = json.load(jsonfile)
            for project in projects:
                repo = project.get("repo")
                status = project.get("status")
                if repo and status in status_mapping:
                    status_data[repo] = status_mapping[status]
                    if "-" in repo:
                        repo_suffix = repo.split("-", 1)[1]
                        status_data[repo_suffix] = status_mapping[status]
    except Exception as e:
        print(f"Error reading JSON: {e}")
    return status_data

def process_large_json(input_filename, output_filename, json_filename, csv_output_filename):
    project_data = defaultdict(lambda: {"users": defaultdict(lambda: {"merged_prs": 0, "closed_prs": 0, "unresolved_prs": 0}),
                                        "project_stats": {"total_prs": 0, "merged_prs": 0, "closed_prs": 0, "unresolved_prs": 0},
                                        "status": None})
    
    status_data = read_json_status(json_filename)
    
    try:
        with open(input_filename, 'r', encoding='utf-8') as infile:
            parser = ijson.items(infile, 'item')
            for pr in parser:
                repo = pr.get("repo")
                user = pr.get("user")
                state = pr.get("state")
                merged_at = pr.get("merged_at")
                
                project_data[repo]["project_stats"]["total_prs"] += 1
                
                # Assign status, handling cases where repo might have a prefix
                status = status_data.get(repo)
                if not status and "-" in repo:
                    repo_suffix = repo.split("-", 1)[1]
                    status = status_data.get(repo_suffix)
                project_data[repo]["status"] = status
                
                if state == "open":
                    project_data[repo]["users"][user]["unresolved_prs"] += 1
                    project_data[repo]["project_stats"]["unresolved_prs"] += 1
                elif state == "closed":
                    if merged_at:
                        project_data[repo]["users"][user]["merged_prs"] += 1
                        project_data[repo]["project_stats"]["merged_prs"] += 1
                    else:
                        project_data[repo]["users"][user]["closed_prs"] += 1
                        project_data[repo]["project_stats"]["closed_prs"] += 1
        
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            json.dump(project_data, outfile, indent=4)
        print(f"Successfully processed PR data and saved to {output_filename}")
        
        write_csv_summary(project_data, csv_output_filename)
    except Exception as e:
        print(f"Error: {e}")

def write_csv_summary(project_data, csv_output_filename):
    try:
        with open(csv_output_filename, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Project Name", "Project State", "Number of Users", "Total PRs", "Merged PRs", "Closed PRs", "Unresolved PRs", "PR Merge %", "PR Reject %", "Merge to Reject Ratio"])
            
            for project, data in project_data.items():
                status = data["status"]
                num_users = len(data["users"])
                total_prs = data["project_stats"]["total_prs"]
                merged_prs = data["project_stats"]["merged_prs"]
                closed_prs = data["project_stats"]["closed_prs"]
                unresolved_prs = data["project_stats"]["unresolved_prs"]
                
                merge_percentage = (merged_prs / total_prs * 100) if total_prs > 0 else 0
                reject_percentage = (closed_prs / total_prs * 100) if total_prs > 0 else 0
                merge_to_reject_ratio = (merged_prs / closed_prs) if closed_prs > 0 else "N/A"
                
                writer.writerow([project, status, num_users, total_prs, merged_prs, closed_prs, unresolved_prs, f"{merge_percentage:.2f}", f"{reject_percentage:.2f}", merge_to_reject_ratio])
        print(f"CSV summary saved to {csv_output_filename}")
    except Exception as e:
        print(f"Error writing CSV summary: {e}")


# Example usage
input_file = "pull-requests.json"  # Change to your actual file name
output_file = "processed_pr_data.json"
json_file = "project-status.json"  # Change to your actual JSON file name
csv_output_file = "pr_summary.csv"
process_large_json(input_file, output_file, json_file, csv_output_file)
