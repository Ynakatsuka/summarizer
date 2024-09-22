# summarizer

summarize the content of the URL using [Gemini API](https://ai.google.dev/pricing?hl=ja).

### Auth

Refer to https://ai.google.dev/palm_docs/oauth_quickstart

```bash
gcloud auth application-default login \
    --client-id-file=client_secret.json \
    --scopes='https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.tuning'
```

### Get Started

Setup Environment

```bash
rye sync
rye run playwright install
```

Summarize the content of the URLs

```bash
rye run python src/summarizer/run.py --urls \
    https://www.google.com \
    https://www.yahoo.co.jp \
    --output output.csv
```

Summarize the content of the PDF files in the GitHub repository

```bash
rye run python src/summarizer/run.py --repo_url https://github.com/tangxyw/RecSysPapers --output recsys_papers.csv
```

Summarize the content of the Kaggle solutions and output to CSV

```bash
rye run python src/summarizer/run.py --kaggle_title otto-recommender-system --output otto_solutions.csv
```

Create a solution matrix for Kaggle solutions

```bash
rye run python src/summarizer/matrix.py --input otto_solutions.csv
```
