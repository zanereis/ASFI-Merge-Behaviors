#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage: $0 directory"
  exit 1
fi

jq '.[] |
  {
    repo: (input_filename | sub(".*/"; "") | sub("^apache-"; "") | sub("-[0-9]*\\.json$"; "")),
    number,
    user: .user.login,
    created_at, updated_at, closed_at, merged_at,
    merge_commit_sha,
    body
  }
' "$1"/*.json | jq -s '.'
