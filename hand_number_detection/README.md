# Hand Number Detection (Beginner-Friendly)

Real-time hand number detection using your laptop camera. Start with a simple baseline, then collect your own data to train a small model.

## 1) Setup

```bash
cd /workspace/hand_number_detection
pip install -r requirements.txt --user --break-system-packages
```

- If you're on a remote environment without a camera, run this project locally on your machine for camera access.
- For best accuracy, install MediaPipe locally (Python 3.10–3.12 recommended): `pip install mediapipe==0.10.14`

## 2) Quick demo

- In this environment (no MediaPipe): OpenCV-only finger counting will run.
- With MediaPipe installed: Uses hand landmarks and optionally your trained model.

```bash
python hand_number.py run
```

- Press `q` to quit.

## 3) Collect your own data (0-5)

Requires MediaPipe for robust landmark extraction and labeling.

```bash
python hand_number.py collect --dataset data/dataset.csv
```

- Show one hand to the camera.
- Press keys `0`, `1`, `2`, `3`, `4`, `5` to save a labeled sample.
- Collect 50-100 samples per class for a decent model (vary angles and distances).
- Press `q` to quit.

## 4) Train a model (lightweight KNN)

This environment trains a simple KNN and saves as NPZ.

```bash
python hand_number.py train --dataset data/dataset.csv --model models/hand_number_model.npz
```

- The model stores the training features and labels; `k` is chosen automatically.
- If you previously trained a scikit-learn model locally, the script can still try to load it (optional).

## 5) Run with your trained model

```bash
python hand_number.py run --model models/hand_number_model.npz
```

You should see predictions and a simple confidence score in the window.

## Notes
- Requirements are minimal here (OpenCV + NumPy). For higher accuracy and stability, install MediaPipe locally and use the `collect` and `run` modes with landmarks.
- If camera cannot open in this environment, run the same commands locally on your machine.