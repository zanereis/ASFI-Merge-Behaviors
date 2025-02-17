#!/bin/bash

input_directory="../data/github-pull-requests"
jq '.[] |
  {
    repo: (input_filename | sub(".*/"; "") | sub("^apache-"; "") | sub("-[0-9]*\\.json$"; "")),
    number,
    user: .user.login,
    user_type: .user.type,
    state,
    created_at, updated_at, closed_at, merged_at,
    merge_commit_sha,
    issue_url,
    body
  }
' "$input_directory"/*.json | jq -s '.'
