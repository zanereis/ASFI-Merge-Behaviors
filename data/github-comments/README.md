This directory contains json containing every pull request for each project in the ASFI. It was populated by running the following command from the tools directory:


```sh
./list-repos.sh ../data/asfi-sustainability-dataset/lists_2019_8.csv | xargs -I {} ./get-comments.sh apache {} ../data/github-comments
```

The previous command requires bash, xargs, and github cli to be installed. Sign into github cli before running.

See more data about the response here:

https://docs.github.com/en/rest/issues/comments?apiVersion=2022-11-28#list-issue-comments-for-a-repository

Notably, this response contains the following:

 - The `body` of the comment.
 - The `issue_url`. This can be used to match the comment to the pull request. Note that according to github, all pull requests are issues while not all issues are pull requests.
