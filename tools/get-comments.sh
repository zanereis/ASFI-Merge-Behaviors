#!/bin/bash

# Get all comments for a given GitHub repo. The comments are saved to a series of files with the format ./OWNER-REPO-number-comments.json
# Usage: ./get-pulls OWNER REPO DIR
# where OWNER is the owner of a GitHub project, REPO is the name of the project, and DIR is where the files are saved. These vales are the same as the ones present in a project's URL.

i=0
prev_output=""

get-pulls() {
	gh api --method GET "https://api.github.com/repos/${owner}/${repo}/issues/comments?per_page=100&page=$i" --header "Accept: application/vnd.github+json"
}

owner="$1"
repo="$2"
dir="$3"

echo "Fetching issue comments from ${owner}/${repo}"

while true; do
	((i++))
	echo "$i"

	output=$(get-pulls 2>/dev/null | tee "$dir"/"${owner}-${repo}-${i}-comments.json")
	status=$?

	if [[ $status -ne 0 ]] ; then
		# gh returned an error, break
		echo "Get pull request command returned non-zero status for ${owner}/${repo} on page ${i}. Status:" $status
		break
	elif [[ "$output" == "$prev_output" ]]; then
		# outputs are identical for the two prevous calls, break
		# remove the duplicate response
		rm "$dir"/"${owner}-${repo}-${i}-comments.json"
		((i--))
		# save the final response so we can inspect it
		mv "$dir"/"${owner}-${repo}-${i}-comments.json" "$dir"/"${owner}-${repo}-FINAL-comments.json"
		break
	fi
	prev_output="$output"

	sleep 0.72  # The GitHub API is rate limited to 5000 calls per hour
done
exit $status
