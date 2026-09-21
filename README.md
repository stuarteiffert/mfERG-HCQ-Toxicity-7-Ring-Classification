# mfERG HCQ Toxicity 7-Ring Classification

This repository contains an interactive machine learning tool to predict Hydroxychloroquine (HCQ) toxicity based on 7-ring P1nv values from multifocal electroretinogram (mfERG) data.

## Features
- **Modern ML Model**: Uses a Random Forest classifier trained on clinical data.
- **Privacy-Focused**: The model runs entirely in your web browser using ONNX Runtime Web. No data is ever sent to a server.
- **Easy Deployment**: Hosted as a static site on GitHub Pages.

## Browser Support
Works in current versions of Chrome, Edge, Firefox and Safari, and on mobile.

The model runs locally via WebAssembly, so:
- **Safari 16.3 and older** lack WebAssembly SIMD. ONNX Runtime Web is therefore
  pinned to **1.18.0**, the last release that also ships a non-SIMD build.
  Newer versions are SIMD-only and fail on those browsers.
- **Safari Lockdown Mode disables WebAssembly entirely.** The calculator cannot
  run with it enabled and will say so on the page.
- The runtime is fetched from the jsDelivr CDN, so networks that block
  `cdn.jsdelivr.net` will prevent the model from loading.

If loading fails, the page shows the reason; the browser console has the detail.

> **Note:** the pinned ONNX Runtime version appears in **both** `index.html`
> (the `<script>` tag) and `script.js` (`ORT_VERSION`). They must always match,
> otherwise the library and the WebAssembly binaries mismatch and fail.

## Local Testing
To run the calculator locally:
1. Navigate to this directory in your terminal.
2. Start a local web server:
   ```bash
   python3 -m http.server 8000
   ```
3. Open your browser and go to `http://localhost:8000`.

*Note: Opening `index.html` directly via `file://` will be blocked by your browser's security policy.*

## Development
- `train.py`: Trains the Random Forest model and saves it to `.joblib`.
- `convert_model.py`: Converts the `.joblib` model and its `StandardScaler` to ONNX and JSON formats for the web.
- `downgrade_model.py`: Lowers the exported model's ONNX IR/opset version so that
  the pinned ONNX Runtime Web build can load it. **Run this after every
  re-export.** skl2onnx emits IR version 10 / opset 22, which ONNX Runtime Web
  1.18.0 rejects. The script verifies predictions are unchanged before saving.

When exporting a new model, keep the web app's expectations intact:
- input `[1, 7]` `float32`, outputs named `label` and `probabilities`
  (export with `options={'zipmap': False}`),
- class `1` means Toxic,
- scaling stays in `scaler.json` — do not bake a `Scaler` into the graph, since
  `script.js` already applies it.

---
*For research purposes only. &copy; 2026 Charles Walker & Stuart Eiffert.*
