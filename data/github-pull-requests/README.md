This directory contains json containing every pull request for each project in the ASFI. It was populated by running the following command from the tools directory:


```sh
./list-repos.sh ../data/asfi-sustainability-dataset/lists_2019_8.csv | xargs -I {} ./get-pulls.sh apache {} ../data/github-pull-requests
```

The previous command requires bash, xargs, and github cli to be installed. Sign into github cli before running.

See more data about the response here:

https://docs.github.com/en/rest/pulls/pulls?apiVersion=2022-11-28


Notably, this response contains:

 - The user who created the pull request
 - The pull request body
 - The number of comments
 - The number of "review comments"
 - A link to an api url that returns the comments on the pull request

