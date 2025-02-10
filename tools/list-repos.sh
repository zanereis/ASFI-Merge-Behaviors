#!/bin/sh
# Reads list_2019_8.csv and returns a list of repos. Each line is the name of an apache github repo that was the ASFI program.
# Usage: ./list-repos.sh ../asfi/lists_2019_8.csv
grep "https://github.com/apache/" "$1" | sed -E 's|.*https://github.com/apache/||' | sed 's/\r$//'
