Render free demo: identify the connection failure

Replace these TWO files in the GitHub repository connected to your Render
Blueprint (your screenshot showed Kelvin-7-art/market-ai):
free_demo/frontend.py
free_demo/backend.py

Do not change render.yaml, dependencies, plans or environment secrets.
Commit both files. Wait for BOTH Render services to finish deploying.
If auto-deploy is off, manually deploy the latest commit on both services.

Frontend > Environment:
BACKEND_URL must be https://market-vision-free.onrender.com
SERVICE_TOKEN must equal the backend's SERVICE_TOKEN. Never share the token.
Save and deploy after changing settings.

Sign in, click Test backend connection and read the result. This test runs
inside Streamlit (not merely your browser) and checks:
1. URL configuration and GET /health.
2. Authenticated GET /ready, verifying matching service tokens.
3. Presence of a backend Groq key, without making a Groq call.
It does not validate the Groq key or guarantee an analysis request succeeds.

The frontend shows the configured backend base URL. Placeholder addresses
are blocked before upload; whitespace and a mistakenly appended /health
are normalized. Connections are HTTPS only; redirects are not followed.
Errors distinguish TLS, connection timeout, read timeout, connection failure,
HTTP 401, HTTP 404 and non-JSON Render responses. No raw secrets are shown.

This is a diagnostic/configuration fix, not proof that the remote cause is
resolved. No Render settings have been changed from here. Send the exact
new error and matching backend log lines if analysis still fails.
