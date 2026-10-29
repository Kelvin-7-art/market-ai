# Deploy the full AI Framework frontend

This is a deployment copy of your uploaded project. It preserves the existing
Dashboard, sidebar, assets, Analytics, model pages, Projects, Registry, Settings,
and chat UI. Your uploaded sidebar brands the app as TradeGPT; that branding is
preserved. It does not deploy or replace the 20-analyzer desktop vision service.

## 1. Update the existing GitHub repository

Extract this ZIP. Upload its CONTENTS into the root of Kelvin-7-art/market-ai,
replacing matching files. app.py must be at the repository root, next to
analytics/, components/, utils/, requirements.txt, packages.txt and market_ai/.
The small market_ai/__init__.py package bridge is required; keep it.
Also upload .streamlit/config.toml (enable hidden files in Explorer if needed).
Do not upload the ZIP itself or put everything inside another patch folder.
Keep your existing free_demo/ and render.yaml files: this package does not
replace the working Render services. Commit the changes. Never upload actual
secrets.toml, .env files or real API keys.

## 2. Create the Streamlit app

Go to https://share.streamlit.io and sign in with GitHub. Choose Create app and
deploy from GitHub. Select Kelvin-7-art/market-ai, your branch (usually main),
and main file path app.py. In Advanced settings select Python 3.12.
The dependency validation for this package uses Python 3.12.
Paste cloud_secrets.example.toml into the Secrets box, replacing placeholders.
CLOUD_MODE must be "true". SERVICE_TOKEN must exactly match market-vision-free.
APP_PASSWORD is the password you will enter to open the full workspace.
GROQ_PRIMARY_API_KEY is for the separate Analytics > Chat AI JSON analyst.
Image/document chat uses the Render /chat endpoint and Render's GROQ_API_KEY.
You do not need to copy Render's Groq key to Streamlit for image chat.
Save and deploy. Never share your completed secrets template.

## 3. Verify after deployment

Sign in, check Dashboard and sidebar, then open Chat (or Analytics > Chat).
Expand Backend connection and click Test backend connection. Upload one chart
under 5 MB, ask a question, then ask a follow-up without uploading it again.
Use New chat to clear the conversation and image cache. Successful analyses
are cached only in the current Streamlit session; failures can be retried.
Test Analytics > Chat AI with a small valid JSON report and Groq Primary.
Test a small Base Model training run before attempting larger workloads.
Yahoo data can still be unavailable or rate-limited independently of deployment.

## Cloud profile and practical limits

- Up to three images per send; each uploaded file is limited to 5 MB. A decoded
  image can produce a PNG larger than the original JPEG; the client checks the
  converted PNG too. Use a smaller/resized image if that limit is exceeded.
- Images and extracted document text are sent to Render/Groq when submitted.
  The chat context is compacted with omission notices; long detail may be lost.
- LSTM and Transformer training remain available in the desktop source but
  are explicitly unavailable in this free cloud profile. PyTorch/Transformers
  are not installed. Sentiment uses the existing labelled finance lexicon
  fallback instead of FinBERT. Do not describe fallback scores as FinBERT output.
- Other models remain available. Basic thread limits and single-worker Random
  Forest jobs reduce CPU pressure, but large tuning/forecast workloads can still
  exceed host resources. This is not a guarantee all training fits free hosting.
- Registry, project files and session history are not durable cloud storage.
  Concurrent users share the server filesystem; this is a password-protected
  research workspace, not a multi-tenant production application. Export results.
- The original desktop 20-analyzer pipeline is not running on Render. The hosted
  client wraps the Groq extraction into the original chat's results structure.
- Keep market-vision-free running. Do not delete the working Render demo until
  the full Streamlit deployment has passed your checks.

## Local use

From this folder: python -m pip install -r requirements.txt
Then: python -m streamlit run app.py
To use the desktop pipeline leave BACKEND_URL unset, set VISION_PIPELINE_URL
if needed (default http://localhost:8000), and set GROQ_API_KEY. Leave CLOUD_MODE
unset/false. Desktop LSTM/Transformer/FinBERT additionally require compatible
PyTorch and Transformers installations; those are intentionally not part of
this cloud dependency file.

## Documentation

https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

The cloud deployment and real Groq credentials must still be verified in your
account. No live Render settings or GitHub files were changed by this package.
