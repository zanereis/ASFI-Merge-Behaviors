`monthly_data.json` contains monthly data about each project in the ASFI. PRs and comments from bots have been removed.

The fields are as follows:

`listid` is the project id as found in the ASFI dataset. This field is identical to the one in `lists_2019_8.csv`.

`repo` is the name of the github repo as it appears in the project's GitHub URL. This is sometimes different from the project's name. Use "listid" if you wish to use this data with the ASFI dataset.

`status` this is the project's status as indicated in the ASFI dataset. This field is populated using data from `lists_2019_8.csv`.

`month` contains a year and a month. This is the year and month the following fields were calculated for.

`response_time` is the average time between comments on a PR. This field is in seconds.

`first_response_time` is the average time it takes for a PR to receive its first comment. This field is in seconds.

`active_devs` is the number of developers who have either created a PR or commented.

`accepted_prs` is the number of previously open PRs that were merged into the repository.

`avg_time_to_acceptance` is the average time it takes for a PR to be merged. This field is in seconds.

`rejected_prs` is the number of previously open PRs that were closed without merging.

`avg_time_to_rejection` is the time it takes for a PR to be closed without merging. This field is in seconds.

`unresolved_prs` is the number of open PRs that were neither accepted nor rejected.

`avg_thread_length` is the average number of comments per open PR. PRs that were open at any time during this month were counted. Comments that were created in subsequent months were not counted. However, comments from previous months were counted.

`new_prs` is the number of newly created PRs.

`new_comments` is the number of newly created comments.
