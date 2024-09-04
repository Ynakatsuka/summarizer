# summarizer

### Auth

Refer to https://ai.google.dev/palm_docs/oauth_quickstart

```bash
gcloud auth application-default login \
    --client-id-file=client_secret.json \
    --scopes='https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/generative-language.tuning'
```

### Get Started

```bash
rye sync
rye run python src/summarizer/app.py --urls \
    https://www.google.com \
    https://www.yahoo.co.jp \
```
