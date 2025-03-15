import csv

input_csv = "pr_summary.csv"
output_csv = "pr_summary_normalized.csv"

# Detect delimiter
with open(input_csv, 'r', encoding='utf-8') as infile:
    sample = infile.readline()
    delimiter = ',' if ',' in sample else '\t'  # Auto-detect delimiter
    print(f"Detected delimiter: '{delimiter}'")

# Read input CSV with correct delimiter
normalized_data = []
with open(input_csv, 'r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile, delimiter=delimiter)  # Use detected delimiter
    headers = reader.fieldnames
    print("Headers found:", repr(headers))  # Debugging

    # Check if the required column exists
    if "Total Active Devs" not in headers:
        raise KeyError(f"Column 'Total Active Devs' is missing! Found: {headers}")

    for row in reader:
        row = {key.strip(): value.strip() for key, value in row.items()}  # Normalize row keys
        
        project_state = int(row.get("Project State", -1))
        if project_state == 0:
            continue  # Skip incubating projects
        
        try:
            total_active_devs = row["Total Active Devs"]
            total_active_devs = int(total_active_devs) if total_active_devs.isdigit() else 0  # Handle 'N/A'
            
            total_prs_norm = float(row["Total PRs"]) / total_active_devs if total_active_devs > 0 else 0
            merged_prs_norm = float(row["Merged PRs"]) / total_active_devs if total_active_devs > 0 else 0
            closed_prs_norm = float(row["Closed PRs"]) / total_active_devs if total_active_devs > 0 else 0
            unresolved_prs_norm = float(row["Unresolved PRs"]) / total_active_devs if total_active_devs > 0 else 0
        except KeyError as e:
            print(f"Missing column: {e}, skipping row: {row}")
            continue  # Skip row if key error occurs
        except ValueError as e:
            print(f"Value error: {e}, skipping row: {row}")
            continue  # Skip if conversion fails

        normalized_data.append([
            row["Project Name"], row["List Id"], project_state,
            round(total_prs_norm, 2), round(merged_prs_norm, 2),
            round(closed_prs_norm, 2), round(unresolved_prs_norm, 2),
            total_active_devs
        ])

# Write output CSV with correct delimiter
with open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=delimiter)
    writer.writerow([
        "Project Name", "List Id", "Project State",
        "Total PRs (normalized)", "Merged PRs (normalized)",
        "Closed PRs (normalized)", "Unresolved PRs (normalized)",
        "Total Active Devs"
    ])
    writer.writerows(normalized_data)

print(f"✅ Normalized PR summary saved to {output_csv}")
