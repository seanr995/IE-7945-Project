# USAJOBS API setup

USAJOBS is an approved public source but its Search API requires a free API key.

1. Request a key at https://developer.usajobs.gov/APIRequest/Index (you provide your
   email address; the key is emailed to you).
2. Add both values to the project `.env` file (never commit it):

```
USAJOBS_EMAIL=you@northeastern.edu
USAJOBS_API_KEY=your-authorization-key
```

   `USAJOBS_EMAIL` is sent as the `User-Agent` header and `USAJOBS_API_KEY` as the
   `Authorization-Key` header, as required by https://developer.usajobs.gov/.
3. Run `python download_data.py` (or `python run_pipeline.py`). The downloader pages
   through `https://data.usajobs.gov/api/search` (500 results/page) and stores every raw
   page under `data/raw/jobs/usajobs/retrieval_<timestamp>/page_NNN.json`.
