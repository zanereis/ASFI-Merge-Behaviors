This directory contains json containing every pull request for each project in the ASFI. It was populated by running the following command from the tools directory:

```sh
./list-repos.sh ../asfi/lists_2019_8.csv | xargs -I {} ./get-pulls.sh apache {} ../data/github-pull-requests
```
