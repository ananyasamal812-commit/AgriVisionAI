# Vercel frontend + Hugging Face API

> Superseded architecture: the current deployment is browser-based and needs only
> Vercel. Follow [DEPLOYMENT_BROWSER.md](DEPLOYMENT_BROWSER.md). The Hugging Face
> Docker creation attempt returned HTTP 402 requiring PRO; the earlier free
> backend assumption below no longer applies. No paid plan was enabled.

## 1. Export a clean deployment

```powershell
.\venv\Scripts\python.exe prepare_deployment.py
```

Unzip `outputs/deployment/huggingface-api.zip` and `vercel-frontend.zip` into
separate local folders. These archives exclude virtual environments, datasets,
training outputs, uploaded photos, and generated reports.

## 2. Hugging Face

Create a **Docker Space** with free CPU hardware. Upload the extracted backend
files, retaining their folders; upload the contents, not the ZIP itself.
The Space README already declares port 7860. The Dockerfile installs CPU-only PyTorch.
The supplied checkpoint is included; you do not need to train on the hosting service.

Wait for `/health` on your Space's direct `https://USERNAME-SPACE.hf.space` URL
to return `status: ok`. A public Space exposes uploaded source and checkpoint files;
confirm your dataset/model redistribution rights before publishing.

## 3. Vercel

Import the extracted frontend into a Git repository and import that repository
in Vercel, or import the full project with **Root Directory = frontend**.
Select the **Vite** preset, build command `npm run build`, output `dist`.
Set this environment variable before building:

```
VITE_API_BASE_URL=https://USERNAME-SPACE.hf.space
```

Use the exact direct Space URL, without `/api`. This public variable contains no secret.
Redeploy the frontend whenever this value changes.

## 4. Connect the origins

In Hugging Face Space Settings, add:

```
ALLOWED_ORIGINS=https://YOUR-PROJECT.vercel.app
ENABLE_EVALUATION=false
```

Use your real Vercel origin without a trailing slash. Add a custom domain or
specific preview origin as a comma-separated value if required. Restart the Space
after changing its variables. Do not use a blanket wildcard for preview domains.

## 5. Verify the live app

Upload a supported leaf JPEG, PNG, or WebP. Confirm the top three matches, heatmap,
original-image toggle, and PDF download. Test a corrupt image and a large upload.
Check the app on a phone. Free Spaces may sleep; the UI allows extra time for a
cold start and offers retry on failure. There is no background keep-alive polling.
Switch to Hindi and download a report; confirm Hindi text renders correctly.
The backend archive includes the open-source font, its license, translations,
and the HarfBuzz dependency required for joined Devanagari text.

Reports/heatmaps expire after 24 hours and may disappear on restart. Original
uploads are deleted after processing. The single-worker API serializes inference;
busy requests return a retry message. This is suitable for a small demonstration,
not a promise of high-traffic availability or field diagnostic accuracy.

## Model training and evaluation

The existing checkpoint remains the serving model. Use the reproducible workflow:

```powershell
.\venv\Scripts\python.exe model_workflow.py audit --output outputs/data-audit
.\venv\Scripts\python.exe model_workflow.py evaluate --split test --output outputs/baseline
.\venv\Scripts\python.exe model_workflow.py train --epochs 3 --output outputs/candidate-run
```

Training uses the train split; selection uses validation, never test. Candidates
store class order, input size, and seed and never overwrite the deployed checkpoint.
Use `--max-per-class 4 --epochs 1` only for a pipeline smoke test. Such a run does
not establish improved accuracy. The current source dataset is PlantVillage;
internet/field robustness needs labeled representative photos and a separate
evaluation set with no near-duplicate overlap. Exact-hash auditing does not find
all near duplicates. Review candidate metrics and field performance before
replacing `checkpoints/best_model.pth` and rebuilding the backend.

The uncertainty thresholds (top score under 70%, or top-two margin under 15 points)
and photo-quality warnings are conservative UI heuristics, not calibrated
out-of-distribution or non-leaf detection. No claim is made that every photo is supported.

Platform references: [Vite on Vercel](https://vercel.com/docs/frameworks/frontend/vite),
[Hugging Face Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).
