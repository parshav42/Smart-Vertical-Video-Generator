from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import os, threading, uuid, time
import cv2 as cv
import numpy as np

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}}, expose_headers=["Content-Disposition", "Content-Length"])

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

jobs = {}  # job_id -> { status, progress, message, output }

def process_video(job_id, input_path, output_path):
    try:
        from ultralytics import YOLO
        import supervision as sv

        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = 5
        jobs[job_id]["message"] = "Loading AI model..."

        model = YOLO("yolov8n.pt")

        cap = cv.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception("Cannot open video")

        fps          = int(cap.get(cv.CAP_PROP_FPS)) or 30
        SRC_W        = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        SRC_H        = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        OUT_W, OUT_H = 1080, 1920
        fourcc = cv.VideoWriter_fourcc(*"XVID")
        temp_out = output_path.replace(".mp4", "_tmp.mp4")
        out     = cv.VideoWriter(temp_out, fourcc, fps, (OUT_W, OUT_H))

        tracker = sv.ByteTrack(
            lost_track_buffer=40,
            minimum_matching_threshold=0.75,
            minimum_consecutive_frames=1
        )

        CONF             = 0.50
        MIN_AREA_RATIO   = 0.02
        MIN_HEIGHT_RATIO = 0.22
        CENTER_SMOOTH    = 0.18
        DEADZONE         = 3
        smooth_x         = None

        def smooth_value(current, target, alpha):
            delta = target - current
            if abs(delta) < DEADZONE:
                return target
            return current + delta * alpha

        def resize_blur_bg(img):
            h, w = img.shape[:2]
            bg    = cv.resize(img, (OUT_W, OUT_H))
            small = cv.resize(bg, (320, 568))
            small = cv.GaussianBlur(small, (301, 301), 0)
            bg    = cv.resize(small, (OUT_W, OUT_H))
            scale = min(OUT_W / w, OUT_H / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv.resize(img, (nw, nh))
            x = (OUT_W - nw) // 2
            y = (OUT_H - nh) // 2
            bg[y:y+nh, x:x+nw] = resized
            return bg

        def valid_person(box, fw, fh):
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            if (bw * bh) / (fw * fh) < MIN_AREA_RATIO: return False
            if bh / fh < MIN_HEIGHT_RATIO:              return False
            return True

        def score_person(box, fw):
            x1, y1, x2, y2 = box
            area  = (x2 - x1) * (y2 - y1)
            cx    = (x1 + x2) / 2
            dist  = abs(cx - fw / 2)
            center_score = 1.0 - (dist / (fw / 2))
            return area * 0.8 + center_score * 100000

        def smart_crop(frame, center_x, crop_w):
            h, w  = frame.shape[:2]
            left  = int(center_x - crop_w / 2)
            right = int(center_x + crop_w / 2)
            if left < 0:  left, right = 0, crop_w
            if right > w: right, left = w, w - crop_w
            return resize_blur_bg(frame[:, left:right])

        frame_count = 0
        jobs[job_id]["progress"] = 10
        jobs[job_id]["message"]  = "Processing frames..."

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            crop_w = min(int(frame_h * 9 / 16), frame_w)

            results    = model(frame, imgsz=640, conf=CONF, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.class_id == 0]
            detections = tracker.update_with_detections(detections=detections)

            persons = []
            for i in range(len(detections)):
                box = detections.xyxy[i].astype(int)
                if not valid_person(box, frame_w, frame_h):
                    continue
                persons.append({"box": box, "score": score_person(box, frame_w)})

            persons.sort(key=lambda p: p["score"], reverse=True)

            if len(persons) == 0:
                final = resize_blur_bg(frame)

            elif len(persons) == 1:
                box = persons[0]["box"]
                x1, y1, x2, y2 = box
                target_x = (x1 + x2) // 2
                if smooth_x is None: smooth_x = target_x
                smooth_x = smooth_value(smooth_x, target_x, CENTER_SMOOTH)
                final = smart_crop(frame, smooth_x, crop_w)

            else:
                box1, box2 = persons[0]["box"], persons[1]["box"]
                combined_x1     = min(box1[0], box2[0])
                combined_x2     = max(box1[2], box2[2])
                combined_center = (combined_x1 + combined_x2) // 2
                PADDING = int(crop_w * 0.05)
                if (combined_x2 - combined_x1) + PADDING * 2 <= crop_w:
                    target_x = combined_center
                    if smooth_x is None: smooth_x = target_x
                    smooth_x = smooth_value(smooth_x, target_x, CENTER_SMOOTH)
                    final = smart_crop(frame, smooth_x, crop_w)
                else:
                    final = resize_blur_bg(frame)

            out.write(final)
            frame_count += 1

            if total_frames > 0:
                pct = 10 + int((frame_count / total_frames) * 75)
                jobs[job_id]["progress"] = pct
                jobs[job_id]["message"]  = f"Processing frame {frame_count}/{total_frames}..."

        cap.release()
        out.release()

        jobs[job_id]["progress"] = 88
        jobs[job_id]["message"]  = "Merging audio..."

        final_out = output_path

        cmd = (
            f'ffmpeg -y '
            f'-i "{temp_out}" '
            f'-i "{input_path}" '
            f'-map 0:v:0 '
            f'-map 1:a? '
            f'-c:v libx264 '
            f'-profile:v high '
            f'-level 4.1 '
            f'-pix_fmt yuv420p '
            f'-preset medium '
            f'-movflags +faststart '
            f'-c:a aac '
            f'-b:a 192k '
            f'"{final_out}"'
        )

        ret = os.system(cmd)

        if ret != 0:
            raise Exception("FFmpeg conversion failed")
        if ret != 0 or not os.path.exists(final_out):
            os.rename(temp_out, final_out)
        elif os.path.exists(temp_out):
            os.remove(temp_out)

        jobs[job_id]["status"]   = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["message"]  = "Complete!"
        jobs[job_id]["output"]   = os.path.basename(final_out)

    except Exception as e:
        jobs[job_id]["status"]  = "error"
        jobs[job_id]["message"] = str(e)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400

    f        = request.files["video"]
    job_id   = str(uuid.uuid4())[:8]
    ext      = os.path.splitext(f.filename)[1] or ".mp4"
    in_path  = os.path.join(UPLOAD_FOLDER, f"{job_id}{ext}")
    out_path = os.path.join(OUTPUT_FOLDER, f"{job_id}_reframed.mp4")

    f.save(in_path)

    jobs[job_id] = {
        "status": "queued", "progress": 0,
        "message": "Queued...", "output": None
    }

    t = threading.Thread(target=process_video, args=(job_id, in_path, out_path), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404
    return jsonify(jobs[job_id])


# FIX: proper byte-range streaming so <video> can seek/play inline
@app.route("/download/<filename>")
def download(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    file_size = os.path.getsize(file_path)
    range_header = request.headers.get("Range", None)

    if range_header:
        # Parse "bytes=start-end"
        byte_start, byte_end = 0, None
        match = range_header.replace("bytes=", "").split("-")
        byte_start = int(match[0])
        byte_end   = int(match[1]) if match[1] else file_size - 1
        byte_end   = min(byte_end, file_size - 1)
        length     = byte_end - byte_start + 1

        def generate_chunk():
            with open(file_path, "rb") as f:
                f.seek(byte_start)
                remaining = length
                chunk_size = 65536
                while remaining > 0:
                    data = f.read(min(chunk_size, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        response = Response(
            generate_chunk(),
            status=206,
            mimetype="video/mp4",
            direct_passthrough=True,
        )
        response.headers["Content-Range"]  = f"bytes {byte_start}-{byte_end}/{file_size}"
        response.headers["Accept-Ranges"]  = "bytes"
        response.headers["Content-Length"] = str(length)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    # Full file
    response = Response(
        open(file_path, "rb").read(),
        status=200,
        mimetype="video/mp4",
    )
    response.headers["Accept-Ranges"]  = "bytes"
    response.headers["Content-Length"] = str(file_size)
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


if __name__ == "__main__":
    print("\n  AI AUTO REFRAME SERVER")
    print("  Open → http://localhost:5000\n")
    app.run(debug=False, port=5000)