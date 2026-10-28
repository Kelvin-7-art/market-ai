# Switch the repository to the Render Free demo

This is a PATCH for your existing repository. Upload free_demo/ and REPLACE
the existing root render.yaml with this version. Leave market_ai/ and
vision_pipeline/ in the repository: the free build does not import or install
them. The hosted demo has a separate lightweight UI for image explanations,
JSON interpretation and follow-ups. It does not host model training, FinBERT,
or the 20 local analyzers. Original local projects remain usable separately.

## GitHub

Open Kelvin-7-art/ai-market-assistant. Upload the free_demo folder, replace
render.yaml at the repository root, and commit. Verify BOTH plan entries
say free. Do not upload the ZIP or any secret keys.

## Render

If the paid Blueprint was never deployed, return to New > Blueprint and
connect the repo again after the commit. Give the Blueprint a new name such
as market-ai-free-demo. Confirm the preview lists only market-vision-free
and market-ai-free, BOTH on Free, with NO disk.

If any paid services were already created, this YAML does NOT delete them.
Stop and remove those separately in Render after checking their names;
changing a Blueprint does not automatically end existing paid services.

Enter:

Backend market-vision-free:
- GROQ_API_KEY: your new/rotated Groq key (not an exposed old key).
- SERVICE_TOKEN: a long random secret you choose, at least 32 characters.
- GROQ_MODEL is already qwen/qwen3.8-27b.

Frontend market-ai-free:
- SERVICE_TOKEN: EXACTLY the same value as the backend.
- APP_PASSWORD: a different strong password for signing into the app.
- BACKEND_URL: https://example.invalid temporarily if Render requires an
  initial value. This deliberately does not target any real service.

Deploy. Once the backend is live, copy its actual public HTTPS onrender.com
URL from its dashboard. Open frontend > Environment, replace BACKEND_URL
with that URL (no /health suffix), and Save/Rebuild/Deploy as offered.
Do not use localhost or the private/internal address.

Open BACKEND_URL/health in a browser: expect status ok and configured true.
Then open the frontend URL, sign in, upload one small chart and ask a question.
Ask a follow-up to confirm the cached extraction is reused. If backend is
sleeping, wait about a minute before retrying. Do not repeatedly submit.

## Limits

Render free service compute has no recurring compute charge, but usage
limits and other account charges may apply. The two services SHARE 750
instance hours per workspace per month; this is not 750 hours each. Both
running continuously would exceed that allowance. Services sleep after
15 minutes idle; do not add keep-alive pings to defeat sleeping.
Build/bandwidth allowances and Groq quotas apply independently. Review
billing limits before deploying if your account has a payment method.

The demo supports 3 images per conversation, 5 MB per image. Images are
sent separately to Groq to extract compact evidence, then text context is
sent to produce a combined answer. Sequential image calls may still exhaust
Groq's per-minute token allowance: successful extractions remain cached,
so wait and retry the question rather than resetting. Reset loses cache.
Sessions and uploads are not persistent. Images are resized to at most
1568 pixels per side for transfer; fine print may need a crop.

Changing attachments or JSON clears conversation history. Large context
is rejected locally; no silent loss of chart series is introduced. Long
model prediction arrays are explicitly omitted from JSON context. Ask
row-level questions using a small relevant excerpt. Recent chat history
is bounded; long or old messages may not reach the LLM.

The API requires SERVICE_TOKEN. The public UI requires APP_PASSWORD.
This shared-password demo is not a production multi-user auth system.
No local model downloads, persistent disks, private services or paid
compute plans are configured. Groq is an external service with its own
billing and rate limits; a free Render instance does not make Groq unlimited.

## Validation

See final delivery message for test results. Live Render deployment and
real Groq calls require your account and have not been performed here.

Official references:
https://render.com/docs/free
https://render.com/docs/infrastructure-as-code
https://console.groq.com/docs/vision
