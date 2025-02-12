#!/bin/bash
if [ -z "$1" ]; then
  echo "Usage: $0 file"
  exit 1
fi
file="$1"
base="$(basename "$file")"
jq -r '(.[0] | keys_unsorted) as $keys | $keys, map([.[ $keys[] ]])[] | @csv' $file
