import argparse
import os
import time
import csv
from typing import List, Tuple, Optional

import cv2
import numpy as np

try:
	import mediapipe as mp
except ImportError:
	mp = None


def ensure_dirs(path: str) -> None:
	dirname = os.path.dirname(path)
	if dirname and not os.path.exists(dirname):
		os.makedirs(dirname, exist_ok=True)


def get_mediapipe_objects():
	if mp is None:
		raise RuntimeError("mediapipe is not installed. Please run: pip install -r requirements.txt")
	mp_hands = mp.solutions.hands
	drawing_utils = mp.solutions.drawing_utils
	drawing_styles = mp.solutions.drawing_styles
	return mp_hands, drawing_utils, drawing_styles


def extract_normalized_landmark_features(hand_landmarks, image_width: int, image_height: int) -> np.ndarray:
	# Use normalized coordinates (0-1), relative to wrist, scaled by max radius
	landmarks_xy: List[Tuple[float, float]] = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
	wrist_x, wrist_y = landmarks_xy[0]
	relative = [(x - wrist_x, y - wrist_y) for (x, y) in landmarks_xy]
	scales = [np.hypot(dx, dy) for (dx, dy) in relative]
	scale = max(max(scales), 1e-6)
	normalized = [(dx / scale, dy / scale) for (dx, dy) in relative]
	flat = []
	for (nx, ny) in normalized:
		flat.extend([nx, ny])
	return np.array(flat, dtype=np.float32)


def count_fingers_rule(hand_landmarks) -> int:
	# Simple heuristic-based finger counting. Assumes roughly upright hand orientation.
	# Landmarks index mapping per MediaPipe:
	# Thumb: 1,2,3,4 | Index: 5,6,7,8 | Middle: 9,10,11,12 | Ring: 13,14,15,16 | Pinky: 17,18,19,20
	lms = hand_landmarks.landmark
	# y increases downward in image coords. A finger is considered extended if tip y < pip y
	def is_extended(tip_idx: int, pip_idx: int) -> bool:
		return lms[tip_idx].y < lms[pip_idx].y
	# Thumb requires special handling: compare x depending on handedness and orientation
	# Use a rough proxy: if tip x is less than ip x for right hand (assuming frontal view)
	# Determine handedness by comparing x of index_mcp and pinky_mcp
	index_mcp_x = lms[5].x
	pinky_mcp_x = lms[17].x
	is_right = index_mcp_x < pinky_mcp_x
	thumb_extended = False
	if is_right:
		thumb_extended = lms[4].x < lms[3].x
	else:
		thumb_extended = lms[4].x > lms[3].x
	index_extended = is_extended(8, 6)
	middle_extended = is_extended(12, 10)
	ring_extended = is_extended(16, 14)
	pinky_extended = is_extended(20, 18)
	return int(thumb_extended) + int(index_extended) + int(middle_extended) + int(ring_extended) + int(pinky_extended)


def draw_hand_annotations(image: np.ndarray, hand_landmarks) -> None:
	mp_hands, drawing_utils, drawing_styles = get_mediapipe_objects()
	drawing_utils.draw_landmarks(
		image,
		hand_landmarks,
		mp_hands.HAND_CONNECTIONS,
		drawing_styles.get_default_hand_landmarks_style(),
		drawing_styles.get_default_hand_connections_style(),
	)


def append_sample_to_csv(dataset_path: str, features: np.ndarray, label: int) -> None:
	ensure_dirs(dataset_path)
	row = list(features.astype(float)) + [int(label)]
	write_header = not os.path.exists(dataset_path)
	with open(dataset_path, mode="a", newline="") as f:
		writer = csv.writer(f)
		if write_header:
			feature_headers = [f"f{i}" for i in range(len(features))]
			writer.writerow(feature_headers + ["label"])
		writer.writerow(row)


def collect_mode(args) -> None:
	mp_hands, drawing_utils, drawing_styles = get_mediapipe_objects()
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise RuntimeError("Could not open webcam (cv2.VideoCapture(0) failed). Ensure a camera is available.")
	with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
		last_save_time = 0.0
		print("Collecting samples. Press keys 0-5 to save a sample for that label. Press q to quit.")
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image.flags.writeable = False
			results = hands.process(image)
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			if results.multi_hand_landmarks:
				for hand_landmarks in results.multi_hand_landmarks:
					draw_hand_annotations(image, hand_landmarks)
					features = extract_normalized_landmark_features(hand_landmarks, image.shape[1], image.shape[0])
					cv2.putText(image, "Press 0-5 to save sample", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
			else:
				cv2.putText(image, "Show your hand to the camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
			cv2.imshow("Collect - Hand Number Dataset", image)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
			if results.multi_hand_landmarks and key in [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
				now = time.time()
				if now - last_save_time > 0.2:  # debounce
					label = int(chr(key))
					features = extract_normalized_landmark_features(results.multi_hand_landmarks[0], image.shape[1], image.shape[0])
					append_sample_to_csv(args.dataset, features, label)
					print(f"Saved sample with label {label}")
					last_save_time = now
	cap.release()
	cv2.destroyAllWindows()


def train_mode(args) -> None:
	dataset_path = args.dataset
	model_path = args.model
	if not os.path.exists(dataset_path):
		raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run collect mode first.")
	# Load CSV manually to avoid heavy deps
	with open(dataset_path, "r") as f:
		reader = csv.reader(f)
		headers = next(reader)
		feature_indices = [i for i, h in enumerate(headers) if h != "label"]
		label_index = headers.index("label") if "label" in headers else len(headers) - 1
		features_list: List[List[float]] = []
		labels_list: List[int] = []
		for row in reader:
			if not row:
				continue
			features_list.append([float(row[i]) for i in feature_indices])
			labels_list.append(int(float(row[label_index])))
	X = np.asarray(features_list, dtype=np.float32)
	y = np.asarray(labels_list, dtype=np.int32)
	# simple holdout validation
	num_samples = X.shape[0]
	perm = np.random.RandomState(42).permutation(num_samples)
	split = max(1, int(0.8 * num_samples))
	train_idx, val_idx = perm[:split], perm[split:]
	X_train, y_train = X[train_idx], y[train_idx]
	X_val, y_val = X[val_idx], y[val_idx] if val_idx.size > 0 else (np.empty((0, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.int32))
	# KNN 'training': store X_train, y_train
	k = 5 if X_train.shape[0] >= 5 else max(1, X_train.shape[0])
	# Evaluate on val
	val_acc = None
	if X_val.shape[0] > 0:
		preds = []
		for v in X_val:
			dists = np.linalg.norm(X_train - v, axis=1)
			idx = np.argsort(dists)[:k]
			votes = y_train[idx]
			labels, counts = np.unique(votes, return_counts=True)
			preds.append(int(labels[np.argmax(counts)]))
		val_acc = float(np.mean(np.asarray(preds) == y_val))
	# Save model as NPZ
	ensure_dirs(model_path)
	if not model_path.endswith(".npz"):
		model_path = os.path.splitext(model_path)[0] + ".npz"
	np.savez(model_path, X=X_train, y=y_train, k=np.array([k], dtype=np.int32))
	if val_acc is not None:
		print(f"Model trained (KNN, k={k}). Validation accuracy: {val_acc:.3f}. Saved to {model_path}")
	else:
		print(f"Model trained (KNN, k={k}). Saved to {model_path}")


def load_model_if_exists(model_path: str):
	if os.path.exists(model_path) and model_path.endswith(".npz"):
		try:
			data = np.load(model_path)
			X = data["X"]
			y = data["y"]
			k = int(data["k"][0]) if "k" in data else 5
			return {"type": "knn", "X": X, "y": y, "k": k}
		except Exception:
			return None
	# Try alternate npz path if given as .pkl
	alt_npz = os.path.splitext(model_path)[0] + ".npz"
	if os.path.exists(alt_npz):
		try:
			data = np.load(alt_npz)
			X = data["X"]
			y = data["y"]
			k = int(data["k"][0]) if "k" in data else 5
			return {"type": "knn", "X": X, "y": y, "k": k}
		except Exception:
			pass
	# Try loading legacy sklearn model if joblib is available
	if os.path.exists(model_path):
		try:
			import joblib  # optional
			model = joblib.load(model_path)
			return {"type": "sklearn", "model": model}
		except Exception:
			return None
	return None


def predict_with_bundle(bundle, features: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
	if bundle is None:
		return None, None
	if bundle.get("type") == "knn":
		X = bundle["X"]
		y = bundle["y"]
		k = int(bundle.get("k", 5))
		dists = np.linalg.norm(X - features, axis=1)
		idx = np.argsort(dists)[:k]
		votes = y[idx]
		labels, counts = np.unique(votes, return_counts=True)
		pred = int(labels[np.argmax(counts)])
		conf = float(np.max(counts) / max(1, k))
		return pred, conf
	if bundle.get("type") == "sklearn":
		model = bundle["model"]
		proba = model.predict_proba([features])[0] if hasattr(model, "predict_proba") else None
		pred = int(model.predict([features])[0])
		conf = float(np.max(proba)) if proba is not None else None
		return pred, conf
	return None, None


def opencv_count_fingers(frame: np.ndarray) -> Tuple[int, dict]:
	# Fallback finger counting using skin segmentation and convexity defects
	# Returns (count, debug_info)
	blurred = cv2.GaussianBlur(frame, (7, 7), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# Basic skin color range in HSV (coarse; works under good lighting)
	lower1 = np.array([0, 30, 60], dtype=np.uint8)
	upper1 = np.array([20, 150, 255], dtype=np.uint8)
	lower2 = np.array([160, 30, 60], dtype=np.uint8)
	upper2 = np.array([179, 150, 255], dtype=np.uint8)
	mask1 = cv2.inRange(hsv, lower1, upper1)
	mask2 = cv2.inRange(hsv, lower2, upper2)
	mask = cv2.bitwise_or(mask1, mask2)
	kernel = np.ones((5, 5), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		return 0, {"mask": mask}
	largest = max(contours, key=cv2.contourArea)
	if cv2.contourArea(largest) < 2000:
		return 0, {"mask": mask, "contour": largest}
	hull_indices = cv2.convexHull(largest, returnPoints=False)
	if hull_indices is None or len(hull_indices) < 3:
		return 0, {"mask": mask, "contour": largest}
	defects = cv2.convexityDefects(largest, hull_indices)
	finger_gaps = 0
	if defects is not None:
		for i in range(defects.shape[0]):
			s, e, f, depth = defects[i, 0]
			start = largest[s][0]
			end = largest[e][0]
			far = largest[f][0]
			a = np.linalg.norm(end - start)
			b = np.linalg.norm(far - start)
			c = np.linalg.norm(end - far)
			if b == 0 or c == 0:
				continue
			# Cosine rule for angle at the defect (far point)
			cos_angle = (b**2 + c**2 - a**2) / (2 * b * c)
			cos_angle = np.clip(cos_angle, -1.0, 1.0)
			angle = np.arccos(cos_angle)
			# depth is scaled by 256 in OpenCV
			if angle <= np.deg2rad(90) and depth > 1000:
				finger_gaps += 1
	count = min(finger_gaps + 1, 5)
	return count, {"mask": mask, "contour": largest}


def run_mode(args) -> None:
	bundle = load_model_if_exists(args.model)
	cap = cv2.VideoCapture(0)
	if not cap.isOpened():
		raise RuntimeError("Could not open webcam (cv2.VideoCapture(0) failed). Ensure a camera is available.")
	# Fallback: OpenCV-only pipeline if mediapipe is unavailable
	if mp is None:
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			count, debug = opencv_count_fingers(frame)
			cv2.putText(frame, f"Count: {count} (opencv)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
			cv2.putText(frame, "Tip: For better accuracy, run locally with mediapipe (see README)", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
			cv2.imshow("Hand Number Detection", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
		cap.release()
		cv2.destroyAllWindows()
		return
	# If mediapipe is available, use the landmark-based pipeline (and model if present)
	mp_hands, drawing_utils, drawing_styles = get_mediapipe_objects()
	with mp_hands.Hands(model_complexity=0, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
		while True:
			ret, frame = cap.read()
			if not ret:
				break
			image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			image.flags.writeable = False
			results = hands.process(image)
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			pred_text = "No hand"
			if results.multi_hand_landmarks:
				hand_landmarks = results.multi_hand_landmarks[0]
				draw_hand_annotations(image, hand_landmarks)
				if bundle is not None:
					features = extract_normalized_landmark_features(hand_landmarks, image.shape[1], image.shape[0])
					pred, conf = predict_with_bundle(bundle, features)
					if pred is not None:
						pred_text = f"Pred: {pred}{f' ({conf:.2f})' if conf is not None else ''}"
					else:
						pred_text = "Pred: ?"
				else:
					count = count_fingers_rule(hand_landmarks)
					pred_text = f"Count: {count} (rule)"
			cv2.putText(image, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
			if bundle is None:
				cv2.putText(image, "Tip: Train a model for better accuracy (see README)", (10, image.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
			cv2.imshow("Hand Number Detection", image)
			key = cv2.waitKey(1) & 0xFF
			if key == ord('q'):
				break
	cap.release()
	cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Hand Number Detection: collect, train, and run real-time detection")
	subparsers = parser.add_subparsers(dest="command", required=True)

	p_collect = subparsers.add_parser("collect", help="Collect labeled samples (press 0-5). q to quit.")
	p_collect.add_argument("--dataset", default="data/dataset.csv", help="Path to dataset CSV")
	p_collect.set_defaults(func=collect_mode)

	p_train = subparsers.add_parser("train", help="Train a classifier from collected samples")
	p_train.add_argument("--dataset", default="data/dataset.csv", help="Path to dataset CSV")
	p_train.add_argument("--model", default="models/hand_number_model.pkl", help="Path to save model")
	p_train.set_defaults(func=train_mode)

	p_run = subparsers.add_parser("run", help="Run real-time detection (uses trained model if available, else rule-based)")
	p_run.add_argument("--model", default="models/hand_number_model.pkl", help="Path to trained model file")
	p_run.set_defaults(func=run_mode)

	return parser


def main():
	parser = build_arg_parser()
	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()