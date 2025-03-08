This directory is the home of pull-requests.json, comments.json, and project-status.json. Since pull-requests.json and comments.json exceeds GitHub's file limits, these files must be either downloaded or generated.

To download them:

Use the drive link found on discord. Unpack the files in this directory.

To build them localy:

Follow the directions in the READMEs for data/pull-requests and data/comments to scrape the data. Github is blocking me from uploading the rest of the data, so you have to scrape it yourself. Then run:

```sh
./build-release.sh
```
