# app.py
import os
import io
import tempfile
import logging
import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify, render_template
import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F
from statistics import median
from werkzeug.utils import secure_filename

# -------- CONFIG --------
MODEL_ID = os.environ.get("HF_MODEL_ID", "Hemg/Deepfake-Detection")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# smoothing parameter for per-frame smoothing (exponential moving average)
SMOOTH_ALPHA = float(os.environ.get("SMOOTH_ALPHA", 0.6))
SMOOTH_ALPHA = max(0.0, min(1.0, SMOOTH_ALPHA))

# allowed image extensions for the image endpoint
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# optional max upload size (bytes) - set to None to disable
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 256 * 1024 * 1024))  # 256MB

# -------- LOGGING --------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("hf_deepfake")

# -------- FLASK APP --------
app = Flask(__name__, static_folder="static", template_folder="templates")
if MAX_CONTENT_LENGTH:
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# -------- MODEL LOAD (fail early with clear message) --------
try:
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.to(DEVICE)
    model.eval()
    log.info("Loaded model %s on %s", MODEL_ID, DEVICE)
except Exception as e:
    log.exception("Failed loading model '%s': %s", MODEL_ID, e)
    raise

# -------- FACE CASCADE --------
LOCAL_HAAR = os.path.join(os.path.dirname(__file__), "models", "haarcascade_frontalface_default.xml")
if os.path.exists(LOCAL_HAAR):
    haar_xml = LOCAL_HAAR
else:
    haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(haar_xml)
if face_cascade.empty():
    log.warning("Failed to load cascade from %s; face detection may not work.", haar_xml)

# -------- UTILITIES --------
def is_image_filename(filename: str) -> bool:
    if not filename:
        return False
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_IMAGE_EXTS

def detect_and_crop_face(pil_img):
    """If faces found return a crop (largest face + padding), otherwise return original RGB PIL."""
    arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    if len(faces) == 0:
        return pil_img.convert("RGB")
    faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
    x, y, w, h = faces[0]
    pad = int(0.2 * max(w, h))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(arr.shape[1], x + w + pad); y1 = min(arr.shape[0], y + h + pad)
    crop = arr[y0:y1, x0:x1]
    return Image.fromarray(crop).convert("RGB")

def detect_and_crop_face_strict(pil_img, min_size=64):
    """Return cropped PIL image of the largest detected face. If no face found, return None."""
    arr = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_size, min_size))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    pad = int(0.25 * max(w, h))
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(arr.shape[1], x + w + pad); y1 = min(arr.shape[0], y + h + pad)
    crop = arr[y0:y1, x0:x1]
    return Image.fromarray(crop).convert("RGB")

def model_predict_probs(pil_img):
    """Return dict label->score for a single PIL image."""
    # processor returns tensors on CPU by default; move to DEVICE
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    id2label = getattr(model.config, "id2label", None)
    if id2label is None:
        id2label = {i: f"LABEL_{i}" for i in range(len(probs))}
    return {id2label[i]: float(probs[i]) for i in range(len(probs))}

def get_fake_score_from_probs(probs_dict):
    """
    Heuristic mapping from model label probs -> single fake_score in [0,1].
    Tries to find a 'fake' label. Falls back to reasonable heuristics for 2-class models.
    """
    lower_map = {k.lower(): v for k, v in probs_dict.items()}
    # direct key
    if "fake" in lower_map:
        return lower_map["fake"]
    # variations
    for k in lower_map:
        if "fake" in k or "deepfake" in k or "manipul" in k:  # manipulated/deepfake
            return lower_map[k]
    # binary fallback
    if len(lower_map) == 2:
        # prefer label '1' if present
        for k in lower_map:
            if k.strip() == "1":
                return lower_map[k]
        vals = list(lower_map.values())
        return vals[1]  # assume second label is fake in many finetunes
    # fallback: highest probability (not ideal)
    return max(lower_map.values())

# -------- ROUTES --------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_ID, "device": DEVICE})

@app.route("/predict_image", methods=["POST"])
def predict_image_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    f = request.files["file"]
    filename = secure_filename(f.filename or "")
    # prevent users from uploading videos to image endpoint
    if not is_image_filename(filename):
        # try inspecting content-type as a fallback
        content_type = (f.content_type or "").lower()
        if not content_type.startswith("image/"):
            return jsonify({"error": "uploaded file does not look like an image; use the video test for videos"}), 400

    try:
        pil = Image.open(io.BytesIO(f.read())).convert("RGB")
    except UnidentifiedImageError as e:
        return jsonify({"error": "invalid image", "detail": str(e)}), 400
    except Exception as e:
        return jsonify({"error": "could not read image", "detail": str(e)}), 400

    crop = detect_and_crop_face(pil)
    probs = model_predict_probs(crop)
    fake_score = get_fake_score_from_probs(probs)
    top_results = sorted([{"label": k, "score": v} for k, v in probs.items()], key=lambda x: x["score"], reverse=True)
    verdict = "FAKE" if fake_score >= 0.5 else "REAL"

    # return a unified small schema to match video endpoint expectations
    return jsonify({
        "verdict": verdict,
        "fake_score": float(fake_score),
        "mean_raw": float(fake_score),
        "mean_smoothed": float(fake_score),
        "median_raw": float(fake_score),
        "percent_above_0.5_raw": 1.0 if fake_score >= 0.5 else 0.0,
        "max_raw": float(fake_score),
        "frames": [{"frame_number": 0, "fake_score": float(fake_score), "smoothed_fake_score": float(fake_score), "top_results": top_results}],
        "sampled_frames": 1
    })

@app.route("/predict_video", methods=["POST"])
def predict_video():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400

    try:
        sample_every_n = int(request.form.get("sample_every_n", 15))
        max_samples = int(request.form.get("max_samples", 30))
        threshold = float(request.form.get("threshold", 0.5))
    except Exception:
        return jsonify({"error": "invalid sampling parameters"}), 400

    vid = request.files["file"]
    filename = secure_filename(vid.filename or "uploaded_video")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".mp4")
    tmp_name = tmp.name
    try:
        # save and ensure file is closed before cv2 uses it
        vid.save(tmp_name)
        tmp.close()

        cap = cv2.VideoCapture(tmp_name)
        if not cap.isOpened():
            return jsonify({"error": "could not open video file"}), 400

        frame_index = 0
        sampled = 0
        frame_results = []
        smoothed = None

        # optional: confidence threshold for excluding very-low-confidence frames (tweak as needed)
        conf_thresh = float(os.environ.get("FRAME_CONF_THRESH", 0.15))

        # video dimensions for optional weighting
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        weighted_sum = 0.0
        weight_total = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            if frame_index % sample_every_n != 0:
                continue

            # convert frame to PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            # strict face crop; skip frame if no face detected
            crop = detect_and_crop_face_strict(pil, min_size=64)
            if crop is None:
                # skip without counting towards sampled
                continue

            probs = model_predict_probs(crop)
            fake_score = get_fake_score_from_probs(probs)
            top_score = max(probs.values()) if len(probs) > 0 else 0.0

            # skip frames with near-zero model confidence (optional)
            if top_score < conf_thresh:
                continue

            # update smoothed score (EMA)
            if smoothed is None:
                smoothed = float(fake_score)
            else:
                smoothed = float(SMOOTH_ALPHA * fake_score + (1 - SMOOTH_ALPHA) * smoothed)

            # weighting by face area (normalize by video)
            face_w, face_h = crop.size
            face_area = float(face_w * face_h)
            normalized_weight = 1.0
            if video_width > 0 and video_height > 0:
                normalized_weight = (face_area / (video_width * video_height)) + 0.01

            weighted_sum += fake_score * normalized_weight
            weight_total += normalized_weight

            frame_results.append({
                "frame_number": frame_index,
                "fake_score": float(fake_score),
                "smoothed_fake_score": float(smoothed),
                "top_results": sorted([{"label": k, "score": v} for k, v in probs.items()], key=lambda x: x["score"], reverse=True),
                "face_area": int(face_area),
                "top_confidence": float(top_score)
            })

            sampled += 1
            if sampled >= max_samples:
                break

        cap.release()
    except Exception as e:
        log.exception("Error processing video: %s", e)
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except: pass
        return jsonify({"error": "processing failed", "detail": str(e)}), 500
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except: pass

    if len(frame_results) == 0:
        return jsonify({"error": "no frames sampled, try lowering sample_every_n or upload a different video"}), 400

    raw_scores = [fr["fake_score"] for fr in frame_results]
    smoothed_scores = [fr["smoothed_fake_score"] for fr in frame_results]

    mean_raw = float(np.mean(raw_scores))
    median_raw = float(median(raw_scores))
    max_raw = float(np.max(raw_scores))
    percent_above_raw = float(sum(1 for s in raw_scores if s >= 0.5) / len(raw_scores))

    mean_smoothed = float(np.mean(smoothed_scores))
    max_smoothed = float(np.max(smoothed_scores))
    weighted_mean = float(weighted_sum / weight_total) if weight_total > 0 else float(mean_raw)

    # combined final decision (heuristic) - tuned to use mean_smoothed for stability
    is_fake = (mean_smoothed >= 0.6) or (percent_above_raw >= 0.25) or (max_raw >= 0.95)
    verdict = "FAKE" if is_fake else "REAL"

    return jsonify({
        "verdict": verdict,
        "mean_raw": mean_raw,
        "median_raw": median_raw,
        "max_raw": max_raw,
        "percent_above_0.5_raw": percent_above_raw,
        "mean_smoothed": mean_smoothed,
        "max_smoothed": max_smoothed,
        "weighted_mean": weighted_mean,
        "sampled_frames": len(frame_results),
        "frames": frame_results
    })

if __name__ == "__main__":
    # bind to 0.0.0.0 only if external access required
    app.run(host="0.0.0.0", port=5000, debug=False)