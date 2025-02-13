#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage: $0 directory"
  exit 1
fi

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
' "$1"/*.json | jq -s '.'
