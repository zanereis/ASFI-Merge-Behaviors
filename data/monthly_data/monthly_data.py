import json
from datetime import datetime
from collections import defaultdict
import statistics
from dateutil.relativedelta import relativedelta

# Load data
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def month_of(time):
    return time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

def main():
    projects = load_json('../../release/project-status.json')
    pull_requests = load_json('../../release/pull-requests.json')
    comments = load_json('../../release/comments.json')
    
    print("filtering bots");

    # Filter out bots
    pull_requests = [pr for pr in pull_requests if pr['user_type'] != 'Bot']
    comments = [comment for comment in comments if comment['user_type'] != 'Bot']
    
    # Organize PRs and comments by issue URL and track active users
    issue_data = defaultdict(list)
    issue_closed_at = {}
    active_users_per_month = defaultdict(set)
    active_users = defaultdict(set)
    pr_accepted = defaultdict(int)
    pr_rejected = defaultdict(int)
    pr_new = defaultdict(int)
    accept_time_data = defaultdict(list)
    reject_time_data = defaultdict(list)

    print("processing PRs");

    for pr in pull_requests:
        issue_data[pr["issue_url"].lower()].append((pr["created_at"], "PR", pr["user"], pr["repo"]))
        issue_closed_at[pr["issue_url"].lower()] = pr["closed_at"]
        created_at = datetime.strptime(pr["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        month_key = created_at.strftime("%Y-%m")
        active_users_per_month[(pr["repo"], month_key)].add(pr["user"])
        active_users[pr["repo"]].add(pr["user"])
        pr_new[(pr["repo"], month_key)] += 1
        
        # Count merged PRs
        if pr["merged_at"]:
            merged_at = datetime.strptime(pr["merged_at"], "%Y-%m-%dT%H:%M:%SZ")
            merged_month = merged_at.strftime("%Y-%m")
            pr_accepted[(pr["repo"], merged_month)] += 1
            # time to acceptance
            accept_time_data[(pr["repo"], merged_month)].append((merged_at - created_at).total_seconds());

        # Count closed PRs
        elif pr["closed_at"]:
            closed_at = datetime.strptime(pr["closed_at"], "%Y-%m-%dT%H:%M:%SZ")
            closed_month = closed_at.strftime("%Y-%m")
            pr_rejected[(pr["repo"], closed_month)] += 1
            # time to rejection
            reject_time_data[(pr["repo"], closed_month)].append((closed_at - created_at).total_seconds());

    print("processing comments");

    comment_new = defaultdict(int)

    for cm in comments:
        issue_data[cm["issue_url"].lower()].append((cm["created_at"], "Comment", cm["user"], cm["repo"]))
        created_at = datetime.strptime(cm["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        month_key = created_at.strftime("%Y-%m")
        active_users_per_month[(cm["repo"], month_key)].add(cm["user"])
        active_users[cm["repo"]].add(cm["user"])
        comment_new[(cm["repo"], month_key)] += 1

    print("processing comment latencies");

    # Compute monthly communication latencies
    response_data = defaultdict(list)
    first_response_data = defaultdict(list)
    all_months = set()

    for issue, events in issue_data.items():
        events.sort(key=lambda x: x[0])  # Sort by timestamp
        timestamps = [datetime.strptime(event[0], "%Y-%m-%dT%H:%M:%SZ") for event in events]
        for i in range(1, len(timestamps)):
            latency = (timestamps[i] - timestamps[i-1]).total_seconds() 
            month_key = timestamps[i].strftime("%Y-%m")
            repo = events[i][3]
            all_months.add((repo, month_key))
            if i == 1:
                first_response_data[(repo, month_key)].append(latency)
            else:
                response_data[(repo, month_key)].append(latency)

    # Ensure all months with PR activity are included
    all_months.update(active_users_per_month.keys())
    all_months.update(pr_accepted.keys())
    all_months.update(pr_rejected.keys())
    all_months.update(pr_new.keys())

    print("processing unresolved PRs");

    # Calculate the final month with a timestamp for a repo
    final_month = {}
    # Iterate through each (repo, month) tuple
    for repo, month in all_months:
        month_dt = datetime.strptime(month, "%Y-%m")
        if repo not in final_month or month_dt > final_month[repo]:
            final_month[repo] = month_dt

    # Count unresolved PRs
    pr_unresolved = defaultdict(int)
    for pr in pull_requests:
        created_at = datetime.strptime(pr["created_at"], "%Y-%m-%dT%H:%M:%SZ")
        closed_at = datetime.strptime(pr["closed_at"], "%Y-%m-%dT%H:%M:%SZ") if pr["closed_at"] else final_month[pr["repo"]]
        month = month_of(created_at)  # Start when the PR is created
        last_month = month_of(closed_at) # End when the PR is closed
        while month < last_month: # Loop from the first month to the last month
            month_key = month.strftime("%Y-%m")
            pr_unresolved[(pr["repo"], month_key)] += 1
            month += relativedelta(months=1)  # Move to the next month

    all_months.update(pr_unresolved.keys())

    print("processing thread lengths")

    # Count the number of comments on open tickets
    comment_open = defaultdict(int)
    for issue, events in issue_data.items():
        events.sort(key=lambda x: x[0])  # Sort by timestamp
        repo = events[0][3]
        # a comment is "closed" when the issue is closed
        closed_at = datetime.strptime(issue_closed_at[issue], "%Y-%m-%dT%H:%M:%SZ") \
            if issue in issue_closed_at and issue_closed_at[issue] is not None \
            else final_month[repo]
        for i in range(1, len(events)):
            # get when the comment is created
            created_at = datetime.strptime(events[i][0], "%Y-%m-%dT%H:%M:%SZ")
            month = month_of(created_at)  # Start when the comment is created
            last_month = month_of(closed_at) # End when the comment is no longer visible (the issue is closed)
            while month <= last_month:
                month_key = month.strftime("%Y-%m")
                comment_open[(repo, month_key)] += 1
                month += relativedelta(months=1)  # Move to the next month

    print("formatting output");

    # Prepare final output
    output_data = []
    project_map = {p["repo"].lower(): (p["listid"], p["status"]) for p in projects}

    for (repo, month) in sorted(all_months):
        if repo.lower() in project_map:
            listid, status = project_map[repo.lower()]
            avg_response_time = statistics.mean(response_data[(repo, month)]) if response_data[(repo, month)] else None
            avg_first_response_time = statistics.mean(first_response_data[(repo, month)]) if first_response_data[(repo, month)] else None
            avg_accept_time = statistics.mean(accept_time_data[(repo, month)]) if accept_time_data[(repo, month)] else None
            avg_reject_time = statistics.mean(reject_time_data[(repo, month)]) if reject_time_data[(repo, month)] else None
            unresolved_prs = pr_unresolved.get((repo, month), 0);
            accepted_prs = pr_accepted.get((repo, month), 0);
            rejected_prs = pr_rejected.get((repo, month), 0);
            # The number of comments that were open this month. It equals the
            # number of comments that were left unresolved plus the prs that
            # were resolved (accepted/rejected)
            open_prs = unresolved_prs + accepted_prs + rejected_prs 
            open_comments = comment_open.get((repo, month), 0);
            avg_thread_length = open_comments / open_prs if open_prs != 0 else None;
            output_data.append({
                "listid": listid,
                "repo": repo,
                "status": status,
                "month": month,
                "avg_response_time": avg_response_time,
                "avg_first_response_time": avg_first_response_time,
                "active_devs": len(active_users_per_month.get((repo, month), [])),
                "total_active_devs": len(active_users.get(repo, [])),
                "accepted_prs": accepted_prs,
                "avg_time_to_acceptance": avg_accept_time,
                "rejected_prs": rejected_prs,
                "avg_time_to_rejection": avg_reject_time,
                "unresolved_prs": unresolved_prs,
                "avg_thread_length": avg_thread_length,
                "new_prs": pr_new.get((repo, month), 0),
                "new_comments": comment_new.get((repo, month), 0)
            })

    # Save results to JSON
    with open("monthly_data.json", "w") as out_file:
        json.dump(output_data, out_file, indent=4)

    print("Processing complete. Results saved to monthly_data.json")

   
if __name__ == "__main__":
    main()
