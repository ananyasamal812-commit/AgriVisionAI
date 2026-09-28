# AgriVisionAI

> Current browser app: run `npm run dev`, with no Python API required. See
> `DEPLOYMENT_BROWSER.md`. The API instructions and server-side retention rules
> below apply only to the optional legacy Python service.

Local plant disease detection using a trained EfficientNet model, FastAPI, and React.

## Start the app (Windows PowerShell)

From the project folder, run:

```powershell
.\venv\Scripts\python.exe run.py
```

Open http://127.0.0.1:5173 after the service is ready. Keep the terminal open.
Press Ctrl+C to stop both services. The launcher automatically chooses free ports
if 8000 or 5173 is already occupied. Always open the URL printed by the new launcher.
Running `npm run dev` or `npm start` from the project root starts both services on Windows.

To keep another server running, choose alternate ports:

```powershell
.\venv\Scripts\python.exe run.py --api-port 8010 --web-port 5175
```

Then open http://127.0.0.1:5175.

## First-time setup

Requires Python 3.11 and Node.js compatible with the installed Vite version.

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-api.txt
npm run install:frontend
.\venv\Scripts\python.exe run.py
```

No PowerShell activation or execution-policy change is necessary.

## Run services separately

API:

```powershell
.\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Frontend, in a second terminal:

```powershell
npm run dev:web
```

API docs: http://127.0.0.1:8000/docs. Readiness: http://127.0.0.1:8000/health.
Both `app:app` and `src.api.app:app` use the same application.
The Vite development server proxies API and heatmap requests to port 8000.

## Use

Upload a JPEG, PNG, or WebP leaf image (up to 10 MB and 20 million pixels).
Select **Analyze leaf**. The result includes the predicted class, confidence,
available disease information, and a heatmap of regions that influenced the model.
It also shows the top three model predictions. Select **Download PDF report**
to save the original image, heatmap, ranked predictions, date, and disease information.
Choose **EN** or **हिन्दी** in the header. Reports download in the currently selected
language, including when you switch after analysis. Hindi fonts and shaping support
are bundled for deployment; no online translation API is used.
Model evaluation is internal: use `model_workflow.py` for auditing and evaluation.
Generated PDFs remain in `outputs/reports` for up to 24 hours and then expire.
Uploads are deleted after processing. Generated heatmaps remain in
`outputs/heatmaps` and also expire after 24 hours.

## Model and scope

The included `checkpoints/best_model.pth` and `configs/classes.json` are required.
Inference uses the image size in `configs/config.yaml`, matching the training pipeline.
The model supports the 38 classes in that class file; it does not detect whether
an arbitrary photo is a supported crop. Confidence is not a guarantee of correctness.
Disease information comes from `data/disease_info.json`; missing entries are shown
as unavailable. Model accuracy on new field photos has not been independently evaluated.

`main.py` trains a model; it is not required to start the web application.
Training and data tools require the additional packages in `requirements.txt`.
The unrelated source previously in that file is preserved in
`requirements-legacy-source.txt`.

## Checks

```powershell
npm run build
npm --prefix frontend run lint
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

Integration tests start a temporary API server and use the real checkpoint and
a sample leaf image. They verify predictions, heatmaps, rejected uploads, and cleanup.
Startup diagnostics are written to `outputs/api-test.log`.

`npm run build` creates the frontend production bundle. For the Vercel/Hugging Face
setup, follow `DEPLOYMENT.md` and configure `VITE_API_BASE_URL` before building.
The supplied launcher is for local use.

See `FIELD_TRAINING.md` for the separate broader field-image research workflow.
