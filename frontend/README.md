# AgriVisionAI frontend

React application with local ONNX Runtime Web inference, Grad-CAM explanations,
and English/Hindi PDF reports.

See the [project README](../README.md) for features, model scope and setup.

```bash
npm ci
npm run dev
npm run lint
npm run build
```

The predev/prebuild step copies the installed WebAssembly runtime into public/ort.
Model and font files are included in public/. No backend environment variables
are required for the deployed browser application.
