#!/bin/sh
# Reads list_2019_8.csv and returns a list of repos. Each line is the name of an apache github repo that was the ASFI program.
# Usage: ./list-repos.sh ../asfi/lists_2019_8.csv

# We filter out 'XMLBeans/C++' because the github url is wrong and the forward slash breaks things
grep "https://github.com/apache/" "$1" | sed -E 's|.*https://github.com/apache/||' | sed 's/\r$//' | grep -v "XMLBeans/C++"
