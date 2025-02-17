#!/bin/sh

if [ "$(basename $PWD)" = "release" ] ; then
	cd ..
fi
cd tools || exit 1
pwd

echo Generating project-status.json
python process-project-status.py > ../release/project-status.json && echo done
echo Generating pull-requests.json
./process-pull-requests.sh > ../release/pull-requests.json && echo done
echo Generating comments.json
./process-comments.sh > ../release/comments.json && echo done
echo Generating project-status.csv
./json-to-csv.sh ../release/project-status.json > ../release/project-status.csv && echo done
echo Generating pull-requests.csv
./json-to-csv.sh ../release/pull-requests.json > ../release/pull-requests.csv && echo done
echo Generating comments.csv
./json-to-csv.sh ../release/comments.json > ../release/comments.csv && echo done
