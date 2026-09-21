# mfERG HCQ Toxicity 7-Ring Classification

This repository contains an interactive machine learning tool to predict Hydroxychloroquine (HCQ) toxicity based on 7-ring P1nv values from multifocal electroretinogram (mfERG) data.

## Features
- **Modern ML Model**: Uses a Random Forest classifier trained on clinical data.
- **Privacy-Focused**: The model runs entirely in your web browser using ONNX Runtime Web. No data is ever sent to a server.
- **Easy Deployment**: Hosted as a static site on GitHub Pages.

## Interpreting the Result
The model outputs a probability of toxicity, `P(toxic)`. The page reports a
three-state verdict together with a confidence value and a confidence bar.

| Verdict | `P(toxic)` |
| --- | --- |
| Normal | below 0.30 |
| Uncertain | 0.30 to 0.55 |
| Toxic | above 0.55 |

- **Decision threshold: 0.40**, based on external validation of this model. It is
  set by `TOXIC_THRESHOLD` in `script.js`.
- The **uncertain band** spans `UNCERTAIN_FRACTION` (0.25) of the distance from
  the threshold to each extreme, so both bounds follow automatically if the
  threshold changes.
- **Confidence** scales the distance from the threshold onto 50%-100%: a result
  sitting on the threshold is a coin flip (50%), and `P(toxic)` of 0 or 1 gives
  100%. It never drops below 50%, since below that the other class would be
  reported. Equivalently, "Uncertain" means confidence of 62.5% or less.

> **Confidence is a distance from the decision boundary, not a calibrated risk.**
> "82% confidence" does **not** mean "82% chance of toxicity". The model is an
> uncalibrated random forest, so its probabilities reflect the proportion of
> trees voting for a class.

> **The displayed verdict intentionally differs from the model's own `label`
> output.** That output is a fixed 0.5 argmax baked into the ONNX
> `TreeEnsembleClassifier` and cannot be changed, so the page derives the
> verdict from `P(toxic)` and the threshold above instead. The console logs a
> warning whenever the two disagree.

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
