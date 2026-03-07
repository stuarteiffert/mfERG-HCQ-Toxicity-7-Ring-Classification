# mfERG HCQ Toxicity 7-Ring Classification

This repository contains an interactive machine learning tool to predict Hydroxychloroquine (HCQ) toxicity based on 7-ring P1nv values from multifocal electroretinogram (mfERG) data.

## Features
- **Modern ML Model**: Uses a Random Forest classifier trained on clinical data.
- **Privacy-Focused**: The model runs entirely in your web browser using ONNX Runtime Web. No data is ever sent to a server.
- **Easy Deployment**: Hosted as a static site on GitHub Pages.

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

---
*For research purposes only. &copy; 2026 Charles Walker & Stuart Eiffert.*
