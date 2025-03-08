This directory contains json containing every pull request for each project in the ASFI. It was populated by running the following command from the tools directory:


```sh
./list-repos.sh ../data/asfi-sustainability-dataset/lists_2019_8.csv | xargs -I {} ./get-pulls.sh apache {} ../data/github-pull-requests
```

The previous command requires bash, xargs, and github cli to be installed. Sign into github cli before running.

See more data about the response here:

https://docs.github.com/en/rest/pulls/pulls?apiVersion=2022-11-28

`404-not-found.txt` contains a list of github urls in `lists_2019_8.csv` that returned error 404 Not Found. 
This was determined by running:

```sh
for file in *-FINAL.json ; do grep '"status":"404"' "$file" >/dev/null && echo "$file" ; done | sed -e 's/apache-//' -e 's/-FINAL\.json//'
```
