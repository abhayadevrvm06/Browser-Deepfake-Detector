# IDP: Edge Deepfake Shield

A real-time, edge-based deepfake detection system. This project passively intercepts audio/video streams (like a Zoom call or YouTube video) and flags synthetic media in real-time entirely in the browser using WebGPU and ONNX runtime.

## 🏗️ Project Architecture
We are operating in a monorepo with two parallel tracks:

* **`ml-pipeline/` (Track A - Python):** Data acquisition, preprocessing, PyTorch model training, and exporting to optimized `.onnx` weights.
* **`browser-extension/` (Track B - TypeScript):** The Manifest V3 Chrome extension that captures web media, runs the Web Worker inference loop, and paints the warning UI.

---

## 🚀 Quick Start Guide

### For the Front-End/Extension Team
1. Navigate to the extension folder: `cd browser-extension`
2. Install dependencies: `npm install`
3. Open Chrome and go to `chrome://extensions`.
4. Turn on **Developer Mode** (top right).
5. Click **Load unpacked** and select the `browser-extension` folder.

### For the Machine Learning Team
1. Navigate to the ML folder: `cd ml-pipeline`
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: 
   * Mac/Linux: `source venv/bin/activate`
   * Windows: `venv\Scripts\activate`
4. *(Note: `requirements.txt` will be populated shortly with our PyTorch and OpenCV dependencies).*

---

## 🛑 Version Control Rules
* **DO NOT** commit raw video/audio datasets to Git.
* **DO NOT** commit heavy `.pth` model weights.
* Only commit finalized, optimized `.onnx` models to the `browser-extension/public/models/` folder. Use Google Drive for sharing raw datasets.
