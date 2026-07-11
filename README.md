# Deepfake Detector

A client-server deepfake detection system built with PyTorch, FastAPI, and Vanilla JavaScript.

## 🏗️ Project Architecture
* **`training/`**: PyTorch code meant to be run in Google Colab (T4 GPU) using transfer learning on EfficientNet-B0.
* **`backend/`**: A lightweight FastAPI server for handling inference, designed to be deployed on Hugging Face Spaces (CPU tier).
* **`extension/`**: A Manifest V3 Chrome extension that allows users to right-click images and verify their authenticity via our backend API.

## 🛑 Version Control Rules
* **DO NOT** commit raw image datasets to Git.
* **DO NOT** commit heavy `.pt` model weights.