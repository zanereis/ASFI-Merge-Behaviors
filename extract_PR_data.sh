#!/bin/bash
ASSET_NAME="comments.json" # or "pull-request.json" depending on what you need for your analysis. You would have to run this twice (with changed names ofc) to get both files.
DEST_FOLDER="$HOME/Desktop/ASFI-Merge-Behaviors"
ASSET_URL="https://github.com/zanereis/ASFI-Merge-Behaviors/releases/download/0.0.1/$ASSET_NAME"

mkdir -p "$DEST_FOLDER"

# download
curl -L -o "$DEST_FOLDER/$ASSET_NAME" "$ASSET_URL"
  
