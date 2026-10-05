import json
import hmac
import os
import shutil
import time
import uuid
import secrets
import zipfile
from datetime import date, timedelta
from pathlib import Path
 
import cv2
import numpy as np
import rawpy
from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
from itsdangerous import BadSignature, URLSafeTimedSerializer

from dotenv import load_dotenv
load_dotenv(override=True)
 
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
 
RAW_EXTENSIONS = {".cr2", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB total per request; RAW files are big
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["CHECKLIST_PASSWORD"] = os.environ.get("CHECKLIST_PASSWORD")
app.config["CHECKLIST_COOKIE_SECRET"] = os.environ.get("CHECKLIST_COOKIE_SECRET")
app.config["CHECKLIST_COOKIE_SECURE"] = os.environ.get("CHECKLIST_COOKIE_SECURE", "true").lower() in {"1", "true", "yes"}

MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
TRACKER_PATH = DATA_DIR / "100_days_tracker.json"
CLEANUP_THRESHOLD = 24 * 3600  # 24 hours in seconds
CHECKLIST_COOKIE_NAME = "checklist_device"
CHECKLIST_COOKIE_MAX_AGE = 10 * 365 * 24 * 60 * 60


def checklist_auth_configured():
    return bool(app.config["CHECKLIST_PASSWORD"] and app.config["CHECKLIST_COOKIE_SECRET"])


def checklist_token_serializer():
    return URLSafeTimedSerializer(app.config["CHECKLIST_COOKIE_SECRET"], salt="checklist-device-v1")


def checklist_device_authorized():
    token = request.cookies.get(CHECKLIST_COOKIE_NAME)
    if not token or not checklist_auth_configured():
        return False

    try:
        payload = checklist_token_serializer().loads(token, max_age=CHECKLIST_COOKIE_MAX_AGE)
    except BadSignature:
        return False

    return isinstance(payload, dict) and payload.get("authorized") is True and isinstance(payload.get("device_id"), str)


# ---------------------------------------------------------------------------
# Manifest and cleanup functions
# ---------------------------------------------------------------------------
def load_manifest():
    """Load the output manifest (batch_id -> timestamp mapping)."""
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_manifest(manifest):
    """Save the output manifest to disk."""
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def cleanup_old_batches():
    """Remove batch folders older than CLEANUP_THRESHOLD."""
    manifest = load_manifest()
    current_time = time.time()
    to_remove = []

    for batch_id, timestamp in list(manifest.items()):
        if current_time - timestamp > CLEANUP_THRESHOLD:
            batch_dir = OUTPUT_DIR / batch_id
            if batch_dir.exists():
                try:
                    shutil.rmtree(batch_dir)
                    to_remove.append(batch_id)
                except Exception as e:
                    print(f"Failed to remove batch {batch_id}: {e}")

    # Update manifest by removing cleaned-up batches
    if to_remove:
        for batch_id in to_remove:
            del manifest[batch_id]
        save_manifest(manifest)
 
 
# ---------------------------------------------------------------------------
# Part 3: worker function — does all the actual image processing for one file
# ---------------------------------------------------------------------------
def process_raw_to_hdr(input_path: Path, output_path: Path, num_exposures: int = 5, ev_range: float = 4.0) -> None:
    """
    Decode a RAW file, generate `num_exposures` synthetic exposures spanning
    `ev_range` stops from its linear sensor data, and merge them with
    Mertens exposure fusion into a single tone-mapped JPEG at output_path.
 
    Raises on any decode/processing failure so the caller can report it
    per-file without killing the whole batch.
    """
    # --- decode RAW to linear RGB ---
    with rawpy.imread(str(input_path)) as raw:
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),  # linear output; we apply our own curve below
        )
    linear = rgb16.astype(np.float32) / 65535.0
 
    # --- generate synthetic exposures by scaling the linear data ---
    stops = np.linspace(-ev_range / 2, ev_range / 2, num_exposures)
    exposures_bgr = []
    for ev in stops:
        gain = 2.0 ** ev
        scaled = np.clip(linear * gain, 0.0, 1.0)
        display = np.power(scaled, 1.0 / 2.2)  # display gamma
        frame = (display * 255).astype(np.uint8)
        exposures_bgr.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
 
    # --- merge with Mertens exposure fusion ---
    merger = cv2.createMergeMertens()
    fused = merger.process(exposures_bgr)
    fused = np.clip(fused, 0.0, 1.0)
    result = (fused * 255).astype(np.uint8)
 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])


def process_jpeg_to_hdr(input_path: Path, output_path: Path, num_exposures: int = 5, ev_range: float = 4.0) -> None:
    """
    Load a JPEG file, generate `num_exposures` synthetic exposures by adjusting exposure,
    and merge them with Mertens exposure fusion into a single tone-mapped JPEG at output_path.
 
    Raises on any decode/processing failure so the caller can report it
    per-file without killing the whole batch.
    """
    # --- load JPEG (8-bit BGR, already gamma-corrected) ---
    img_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Failed to load JPEG: {input_path}")
 
    # Convert to float32 in range [0, 1] for processing
    img_float = img_bgr.astype(np.float32) / 255.0
 
    # --- generate synthetic exposures by adjusting brightness ---
    stops = np.linspace(-ev_range / 2, ev_range / 2, num_exposures)
    exposures_bgr = []
    for ev in stops:
        gain = 2.0 ** ev
        scaled = np.clip(img_float * gain, 0.0, 1.0)
        frame = (scaled * 255).astype(np.uint8)
        exposures_bgr.append(frame)
 
    # --- merge with Mertens exposure fusion ---
    merger = cv2.createMergeMertens()
    fused = merger.process(exposures_bgr)
    fused = np.clip(fused, 0.0, 1.0)
    result = (fused * 255).astype(np.uint8)
 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
 
 
# ---------------------------------------------------------------------------
# Part 2: routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/hdr")
def hdr():
    cleanup_old_batches()
    return render_template("hdr.html")


@app.before_request
def require_checklist_api_auth():
    if (
        request.path == "/api/checklist"
        or request.path.startswith("/api/checklist/")
        or request.path == "/static/100_days_tracker.json"
    ):
        if not checklist_auth_configured():
            return jsonify({"error": "Checklist access is not configured"}), 503
        if not checklist_device_authorized():
            return jsonify({"error": "Checklist login required"}), 401


@app.after_request
def prevent_checklist_caching(response):
    if (
        request.path == "/checklist"
        or request.path.startswith("/api/checklist")
        or request.path == "/static/100_days_tracker.json"
    ):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/checklist", methods=["GET", "POST"])
def checklist():
    if not checklist_auth_configured():
        return render_template("checklist_login.html", setup_error=True), 503

    if checklist_device_authorized():
        return render_template("checklist.html")

    if request.method == "POST":
        submitted_password = request.form.get("password", "")
        if hmac.compare_digest(
            submitted_password.encode("utf-8"),
            app.config["CHECKLIST_PASSWORD"].encode("utf-8"),
        ):
            token = checklist_token_serializer().dumps({
                "authorized": True,
                "device_id": secrets.token_urlsafe(32),
            })
            response = redirect(url_for("checklist"))
            response.set_cookie(
                CHECKLIST_COOKIE_NAME,
                token,
                max_age=CHECKLIST_COOKIE_MAX_AGE,
                secure=app.config["CHECKLIST_COOKIE_SECURE"],
                httponly=True,
                samesite="Lax",
                path="/",
            )
            return response

        return render_template("checklist_login.html", login_error=True), 401

    return render_template("checklist_login.html")


@app.route("/api/checklist", methods=["GET"])
def get_checklist():
    try:
        with open(TRACKER_PATH, "r") as tracker_file:
            return jsonify(json.load(tracker_file))
    except (OSError, json.JSONDecodeError):
        return jsonify({"error": "Unable to load checklist data"}), 500


@app.route("/api/checklist/<int:day_number>", methods=["PATCH"])
def update_checklist_day(day_number):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or len(payload) != 1:
        return jsonify({"error": "Provide one checklist field to update"}), 400

    field, value = next(iter(payload.items()))
    if field not in {"exercise", "eat-less", "read"} and field != "notes":
        return jsonify({"error": "Invalid checklist field"}), 400
    if field == "notes":
        if not isinstance(value, str) or len(value) > 20000:
            return jsonify({"error": "Notes must be text under 20,000 characters"}), 400
    elif not isinstance(value, bool):
        return jsonify({"error": "Checklist items must be true or false"}), 400

    try:
        with open(TRACKER_PATH, "r") as tracker_file:
            tracker = json.load(tracker_file)
    except (OSError, json.JSONDecodeError):
        return jsonify({"error": "Unable to load checklist data"}), 500

    day_key = f"day-{day_number}"
    if day_key not in tracker:
        existing_days = [
            int(key[4:])
            for key in tracker
            if key.startswith("day-") and key[4:].isdigit()
        ]
        last_day = max(existing_days, default=0)
        if day_number != last_day + 1:
            return jsonify({"error": "Only the next sequential day can be added"}), 404
        try:
            last_date = date.fromisoformat(tracker[f"day-{last_day}"]["date"])
        except (KeyError, ValueError):
            return jsonify({"error": "Unable to determine the next checklist date"}), 500
        tracker[day_key] = {
            "date": (last_date + timedelta(days=1)).isoformat(),
            "exercise": False,
            "eat-less": False,
            "read": False,
        }

    tracker[day_key][field] = value
    temporary_path = TRACKER_PATH.with_suffix(".tmp")
    try:
        with open(temporary_path, "w") as tracker_file:
            json.dump(tracker, tracker_file, indent=2)
        temporary_path.replace(TRACKER_PATH)
    except OSError:
        return jsonify({"error": "Unable to save checklist data"}), 500

    return jsonify({"day": tracker[day_key]})
 
 
@app.route("/process", methods=["POST"])
def process():
    mode = request.form.get("mode", "raw")
    if mode not in ("raw", "jpeg"):
        return jsonify({"error": "Invalid mode. Supported modes: raw, jpeg"}), 400
 
    try:
        num_exposures = int(request.form.get("num_exposures", 7))
        ev_range = float(request.form.get("ev_range", 3.0))
    except ValueError:
        return jsonify({"error": "Invalid num_exposures or ev_range"}), 400
    num_exposures = max(3, min(num_exposures, 9))
    ev_range = max(1.0, min(ev_range, 8.0))
 
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400
 
    batch_id = uuid.uuid4().hex[:12]
    batch_upload_dir = UPLOAD_DIR / batch_id
    batch_output_dir = OUTPUT_DIR / batch_id
    batch_upload_dir.mkdir(parents=True, exist_ok=True)
    batch_output_dir.mkdir(parents=True, exist_ok=True)
 
    results = []
    for f in files:
        original_name = f.filename or "unknown"
        ext = Path(original_name).suffix.lower()
 
        # Validate file type based on mode
        valid_extensions = RAW_EXTENSIONS if mode == "raw" else JPEG_EXTENSIONS
        if ext not in valid_extensions:
            results.append({"name": original_name, "ok": False, "error": "unsupported file type"})
            continue
 
        safe_stem = Path(original_name).stem.replace("/", "_").replace("\\", "_")
        input_path = batch_upload_dir / f"{safe_stem}{ext}"
        output_name = f"{safe_stem}_hdr.jpg"
        output_path = batch_output_dir / output_name
 
        try:
            f.save(str(input_path))
            if mode == "raw":
                process_raw_to_hdr(input_path, output_path, num_exposures=num_exposures, ev_range=ev_range)
            else:  # jpeg
                process_jpeg_to_hdr(input_path, output_path, num_exposures=num_exposures, ev_range=ev_range)
            results.append({
                "name": original_name,
                "ok": True,
                "download_url": f"/download/{batch_id}/{output_name}",
                "preview_url": f"/download/{batch_id}/{output_name}",
            })
        except Exception as e:
            results.append({"name": original_name, "ok": False, "error": str(e)})
        finally:
            # Clean up uploaded originals
            if input_path.exists():
                input_path.unlink()
 
    response = {"results": results}
 
    succeeded = [r for r in results if r["ok"]]
    if len(succeeded) > 1:
        zip_path = batch_output_dir / "hdr_batch.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for r in succeeded:
                out_file = batch_output_dir / Path(r["download_url"]).name
                zf.write(out_file, arcname=out_file.name)
        response["zip_url"] = f"/download/{batch_id}/hdr_batch.zip"
 
    # Register batch in manifest
    manifest = load_manifest()
    manifest[batch_id] = time.time()
    save_manifest(manifest)

    return jsonify(response)
 
 
@app.route("/download/<batch_id>/<filename>")
def download(batch_id, filename):
    directory = OUTPUT_DIR / batch_id
    return send_from_directory(directory, filename, as_attachment=False)


if __name__ == "__main__":
    # Bind to all interfaces so it's reachable behind Caddy/nginx on the e2-micro
    app.run(host="0.0.0.0", port=5000)