import json
import os
import shutil
import time
import uuid
import zipfile
from pathlib import Path
 
import cv2
import numpy as np
import rawpy
from flask import Flask, jsonify, render_template, request, send_from_directory
 
app = Flask(__name__)
 
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
 
RAW_EXTENSIONS = {".cr2", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
JPEG_EXTENSIONS = {".jpg", ".jpeg"}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB total per request; RAW files are big
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
CLEANUP_THRESHOLD = 24 * 3600  # 24 hours in seconds


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