# Data Directory

Raw data are intentionally not committed.

Use this directory for local, untracked data downloads when reproducing the workflow. The `.gitignore` file keeps raw, intermediate, and processed data products out of version control.

Recommended local layout:

```text
data/
  raw/
  interim/
  processed/
  external/
```

Keep credentials in a private `.env` file or shell environment variables.
