# ESCO manual download required

The official ESCO download requires a human to accept the European Commission
privacy statement and enter an email address; the download link is emailed.
The pipeline does **not** automate or bypass this step.

1. Open the official page: https://esco.ec.europa.eu/en/use-esco/download
2. Version: **ESCO dataset - v1.2.1**
3. Content: **classification**
4. Language: **en**
5. File type: **csv**
6. Accept the privacy statement, enter your email, open the emailed link, and save the ZIP
   (unchanged, any filename ending in `.zip`) into:

   `data/reference/esco/raw/`

Then run `python run_pipeline.py`. The pipeline auto-discovers the ZIP, verifies it,
extracts it to `data/reference/esco/extracted/<version>/`, and loads the official CSVs in
place of the temporary API-derived tables.
