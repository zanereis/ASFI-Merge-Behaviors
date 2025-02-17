#!/bin/bash

input_directory="../data/github-comments"
jq '.[] |
  {
    repo: (input_filename | sub(".*/"; "") | sub("^apache-"; "") | sub("-[0-9]*\\-comments.json$"; "")),
    id,
    user: .user.login,
    user_type: .user.type,
    author_association,
    created_at, updated_at,
    issue_url,
    body
  }
' "$input_directory"/*.json | jq -s '.'
