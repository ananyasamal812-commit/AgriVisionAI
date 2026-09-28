# Current deployment: Vercel only

The public frontend runs the existing model on the visitor's device using ONNX
Runtime Web. Vercel serves static files. No Python server, Hugging Face Space,
API key, or VITE_API_BASE_URL is required. Photos are not uploaded. Predictions,
Grad-CAM heatmaps, and English/Hindi PDF reports are generated locally.

The first analysis downloads approximately 17.9 MB of model weights plus the
WebAssembly runtime. Performance depends on the visitor's device. Save reports
before refreshing: the app does not persist photos/results. Hindi PDF pages are
rendered with browser text shaping and embedded as high-resolution images.

This removes backend sleeping and inference-server costs. Vercel's free-plan
usage limits still apply; uninterrupted availability is not guaranteed.

## Local use

```powershell
npm run install:frontend
npm run dev
```

Open the URL printed by Vite. Python is not required for the web app.

## Deploy

```powershell
npm run build
npx vercel --prod --yes --cwd frontend --scope acme-88b4
```

Project: agrivision-ai. Framework: Vite. Root: frontend. Build: npm run build.
Output: dist. The prebuild script copies WebAssembly files matching the locked
JavaScript package. Deploy the model, metadata, font, and runtime together.
The .vercelignore excludes local environment files and development dependencies
from source upload. Never deploy the repository root, datasets, or Python venv.

## Re-export only when the validated checkpoint changes

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-export.txt
.\venv\Scripts\python.exe export_browser_model.py
node frontend/scripts/test-browser-model.mjs
node frontend/scripts/test-browser-report.mjs
npm run build
npm --prefix frontend run lint
```

The exporter rejects smoke-test checkpoints. Conversion was checked on 15 real
images with 100% leading-class agreement. Maximum WebAssembly/PyTorch logit
difference was approximately 0.000022; the browser preprocessing confidence
difference was at most 0.36 percentage points on those images. Analytic Grad-CAM
at the final EfficientNet block was checked against autograd.
These are conversion checks, not a new evaluation of disease accuracy.

The model still supports 38 conditions across 14 crops. Unsupported leaves can
receive confident wrong predictions. The broader training candidate has not been
promoted; see FIELD_TRAINING.md.

## Optional Python API

The API remains in the source for research/local use. The new public frontend
does not call it. The attempted Hugging Face Docker deployment returned HTTP 402:
compute Spaces require a paid plan. No paid subscription was started.
DEPLOYMENT.md describes the earlier, superseded backend-based architecture.

Sources: [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/),
[Hugging Face compute requirements](https://huggingface.co/docs/hub/spaces-overview).
