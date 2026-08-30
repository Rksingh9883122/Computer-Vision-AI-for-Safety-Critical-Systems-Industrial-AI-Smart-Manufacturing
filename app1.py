from __future__ import annotations
from streamlit_autorefresh import st_autorefresh
import base64
import csv
import json
import logging
import os
import platform
import smtplib
import ssl
import threading
import time
import zipfile
from collections import deque, Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email import encoders
from email.mime.application import MIMEApplication
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import urlparse

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    render_template_string,
    request,
    send_file,
    url_for,
)
from jinja2 import TemplateNotFound
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# ======================================
# Environment & Logging
# ======================================
load_dotenv()
matplotlib.use("Agg")  # headless rendering for servers

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hse-app")

# ======================================
# Paths & Configuration
# ======================================
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"
MODEL_PATH = BASE_DIR / "runs" / "detect" / "train3" / "weights" / "best.pt"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "m4v"}
ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"}

app = Flask(
    __name__,
    static_folder="static",
    static_url_path="/static",
    template_folder="templates",
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "replace-this-with-a-secure-random-string")

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "500"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

MIN_CONF = float(os.environ.get("MIN_CONF", "0.25"))    # CSV log threshold
PRED_CONF = float(os.environ.get("PRED_CONF", "0.25"))  # model prediction conf
FPS_FALLBACK = float(os.environ.get("FPS_FALLBACK", "30.0"))
PRED_DEVICE = os.environ.get("PRED_DEVICE", None)

_whitelist_env = os.environ.get("SAFETY_CLASS_WHITELIST")
SAFETY_CLASS_WHITELIST: Optional[Set[str]] = (
    {s.strip() for s in _whitelist_env.split(",")} if _whitelist_env else None
)

# --------------------------------------
# Real-time metrics config
# --------------------------------------
REALTIME_THRESHOLD_SEC = float(os.environ.get("REALTIME_THRESHOLD_SEC", "2.0"))  # <= 2s is "live"
OVERLAY_TIMING = os.environ.get("OVERLAY_TIMING", "1") == "1"  # draw LIVE/LAG & timings on frames

# --------------------------------------
# Clip Recording Settings
# --------------------------------------
CLIP_ENABLED = os.getenv("CLIP_ENABLED", "1") == "1"
CLIP_PRE_SEC = int(os.getenv("CLIP_PRE_SEC", "5"))
CLIP_POST_SEC = int(os.getenv("CLIP_POST_SEC", "5"))
CLIP_COOLDOWN_SEC = int(os.getenv("CLIP_COOLDOWN_SEC", "60"))
CLIP_USE_ANNOTATED = os.getenv("CLIP_USE_ANNOTATED", "1") == "1"

CLIP_DIR = OUTPUT_DIR / "clips"
CLIP_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------
# Multi-stream background recording
# --------------------------------------
REC_ENABLED = os.getenv("REC_ENABLED", "1") == "1"
REC_SEGMENT_SEC = int(os.getenv("REC_SEGMENT_SEC", "900"))  # 15 min
REC_DIR = Path(os.getenv("REC_DIR", str(OUTPUT_DIR / "recordings")))
REC_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------
# Load YOLO Model Once
# --------------------------------------
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found at: {MODEL_PATH}\n"
        "Place your trained model there or update MODEL_PATH."
    )

model = YOLO(str(MODEL_PATH))
if PRED_DEVICE:
    try:
        model.to(PRED_DEVICE)
    except Exception:
        pass

try:
    model.fuse()
except Exception:
    pass

_model_lock = threading.Lock()  # thread-safe inference lock

# ======================================
# Utilities
# ======================================
def allowed_video_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def allowed_image_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def open_capture(source: Union[str, int]) -> Optional[cv2.VideoCapture]:
    """
    Create a cv2.VideoCapture from webcam index, file path, or URL.
    For RTSP with GStreamer set USE_GSTREAMER=1 (requires gstreamer installed).
    """
    use_gst = os.environ.get("USE_GSTREAMER") == "1"

    if isinstance(source, str) and use_gst and source.startswith(("rtsp://", "rtsps://")):
        gst = (
            f"rtspsrc location={source} latency=100 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! "
            "videoconvert ! appsink drop=1 max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    else:
        if isinstance(source, int):
            if platform.system().lower().startswith("win"):
                # You can also try cv2.CAP_MSMF on Windows if DSHOW is problematic
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap or not cap.isOpened():
        return None

    # Helpful for webcams
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    except Exception:
        pass

    return cap


def _error_image(text: str = "Error") -> np.ndarray:
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return img


def _encode_mjpeg(bgr_frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", bgr_frame)
    if not ok:
        bgr_frame = _error_image("MJPEG encoding failed")
        ok, buffer = cv2.imencode(".jpg", bgr_frame)
    frame = buffer.tobytes()
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


# ======================================
# CSV Logger
# ======================================
class CSVLogger:
    """
    Simple CSV logger for detections (one file per session/stream).
    Includes camera_id and location columns.
    """

    def __init__(self, log_dir: Path, source_label: str, camera_id: str = "", location: str = ""):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_source = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in source_label)[:60]
        self.path = log_dir / f"detections_{safe_source}_{ts}.csv"
        self.file = open(self.path, mode="w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "server_time_iso",
                "source",
                "camera_id",
                "location",
                "frame_index",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )
        self.file.flush()
        self.camera_id = camera_id
        self.location = location

    @staticmethod
    def _safe_cell(val: Any) -> Any:
        # Prevent CSV injection in Excel
        if not isinstance(val, str):
            return val
        if val and val[0] in ("=", "+", "-", "@"):
            return "'" + val
        return val

    def log_detection(
        self,
        server_time_iso: str,
        source: str,
        frame_idx: int,
        class_id: int,
        class_name: str,
        conf: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> None:
        self.writer.writerow(
            [
                self._safe_cell(server_time_iso),
                self._safe_cell(source),
                self._safe_cell(self.camera_id),
                self._safe_cell(self.location),
                frame_idx,
                class_id,
                self._safe_cell(class_name),
                f"{float(conf):.4f}",
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            ]
        )
        self.file.flush()

    def close(self) -> None:
        try:
            self.file.close()
        except Exception:
            pass


# ======================================
# Email Alerts
# ======================================
class EmailAlerter:
    """
    Simple SMTP email sender.
    Env:
      SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASS,
      SMTP_USE_TLS=1|0, SMTP_USE_SSL=1|0, FROM_EMAIL
    """

    def __init__(self):
        self.reload_from_env()

    def reload_from_env(self):
        self.smtp_host = os.getenv("SMTP_HOST", "").strip()
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "").strip()
        self.smtp_pass = os.getenv("SMTP_PASS", "").strip()
        self.use_tls = os.getenv("SMTP_USE_TLS", "1") == "1"
        self.use_ssl = os.getenv("SMTP_USE_SSL", "0") == "1" or self.smtp_port == 465
        self.from_email = os.getenv("FROM_EMAIL", self.smtp_user or "alerts@localhost")

        logger.info(
            "SMTP config host=%s port=%s tls=%s ssl=%s from=%s user_set=%s pass_set=%s",
            self.smtp_host,
            self.smtp_port,
            self.use_tls,
            self.use_ssl,
            self.from_email,
            bool(self.smtp_user),
            bool(self.smtp_pass),
        )

    def send(
        self,
        subject: str,
        html_body: str,
        to_emails: List[str],
        attachments: Optional[List[Tuple[str, bytes, str]]] = None,
    ) -> bool:
        if not self.smtp_host or not to_emails:
            app.logger.warning("Email not sent: SMTP_HOST or recipients missing.")
            return False

        msg = MIMEMultipart("mixed")
        msg["From"] = self.from_email
        msg["To"] = ", ".join(to_emails)
        msg["Date"] = formatdate(localtime=True)
        msg["Subject"] = subject

        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText("This email contains HTML content.", "plain", "utf-8"))
        alt.attach(MIMEText(html_body, "html", "utf-8"))
        msg.attach(alt)

        for (fname, data, mime) in (attachments or []):
            try:
                if mime and "/" in mime:
                    maintype, subtype = mime.split("/", 1)
                    part = MIMEBase(maintype, subtype)
                    part.set_payload(data)
                    encoders.encode_base64(part)
                else:
                    part = MIMEApplication(data, Name=fname)
                part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
                msg.attach(part)
            except Exception:
                app.logger.exception("Failed to attach %s", fname)

        try:
            if self.use_ssl:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=context, timeout=30) as s:
                    s.ehlo()
                    if self.smtp_user:
                        s.login(self.smtp_user, self.smtp_pass)
                    s.send_message(msg)
            else:
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as s:
                    s.ehlo()
                    if self.use_tls:
                        context = ssl.create_default_context()
                        s.starttls(context=context)
                        s.ehlo()
                    if self.smtp_user:
                        s.login(self.smtp_user, self.smtp_pass)
                    s.send_message(msg)
            return True
        except Exception as e:
            app.logger.exception("Failed to send email: %s", e)
            return False


class AlertConfig:
    def __init__(self):
        self.enabled = os.getenv("ALERTS_ENABLED", "1") == "1"
        rec_env = os.getenv("ALERT_RECIPIENTS", "")
        self.recipients = [e.strip() for e in rec_env.split(",") if e.strip()]
        self.cooldown_sec = int(os.getenv("ALERT_COOLDOWN_SEC", "120"))
        neg_env = os.getenv("NEGATIVE_CLASSES", "")
        self.negative_classes = {s.strip().lower() for s in neg_env.split(",") if s.strip()}
        self.last_sent: Dict[Tuple[str, str, str], float] = {}
        self.lock = threading.Lock()

    def update(self, enabled: bool, recipients: List[str], cooldown_sec: int, negative_classes: Set[str]):
        with self.lock:
            self.enabled = enabled
            self.recipients = recipients
            self.cooldown_sec = cooldown_sec
            self.negative_classes = {s.lower() for s in negative_classes}

    def should_alert(self, camera_id: str, location: str, cls_name: str) -> bool:
        key = (camera_id or "", location or "", cls_name or "")
        now = time.time()
        with self.lock:
            last = self.last_sent.get(key, 0.0)
            if now - last >= self.cooldown_sec:
                self.last_sent[key] = now
                return True
            return False


email_alerter = EmailAlerter()
alerts_config = AlertConfig()


def _send_violation_email_async(context: Dict[str, Any], snapshot_bgr: Optional[np.ndarray]) -> None:
    def _work():
        try:
            with app.app_context():
                try:
                    html_body = render_template("email_violation.html", **context)
                except TemplateNotFound:
                    html_body = f"""
                    <html><body>
                      <h3>HSE Alert: {context.get('cls_name','Unknown')}</h3>
                      <ul>
                        <li><b>Time (UTC):</b> {context.get('timestamp_utc','')}</li>
                        <li><b>Camera ID:</b> {context.get('camera_id','-')}</li>
                        <li><b>Location:</b> {context.get('location','-')}</li>
                        <li><b>Source:</b> {context.get('source_label','')}</li>
                        <li><b>Confidence:</b> {context.get('confidence','')}</li>
                        <li><b>Box:</b> ({context.get('x1')},{context.get('y1')}) - ({context.get('x2')},{context.get('y2')})</li>
                      </ul>
                    </body></html>
                    """.strip()

                attachments: List[Tuple[str, bytes, str]] = []
                if snapshot_bgr is not None:
                    ok, buf = cv2.imencode(".jpg", snapshot_bgr)
                    if ok:
                        attachments.append((f"snapshot_{int(time.time())}.jpg", buf.tobytes(), "image/jpeg"))

                subject = f"[HSE Alert] Negative Observation: {context.get('cls_name','Unknown')} @ {context.get('location','Unknown')}"
                recipients = context.get("recipients", [])
                if not recipients:
                    app.logger.warning("No recipients configured; skipping email.")
                    return

                ok_send = email_alerter.send(subject, html_body, recipients, attachments=attachments)
                if not ok_send:
                    app.logger.error("Email send failed: %s", subject)
        except Exception as e:
            app.logger.exception("Email thread error: %s", e)

    threading.Thread(target=_work, daemon=True).start()


# ======================================
# Event Clip Recorder
# ======================================
class EventClipRecorder:
    def __init__(self, out_dir: Path, fps: float, pre_sec: int = 5, post_sec: int = 5, use_annotated: bool = True):
        self.out_dir = out_dir
        self.fps = max(float(fps or 0), 1.0)
        self.pre_sec = max(int(pre_sec), 0)
        self.post_sec = max(int(post_sec), 0)
        self.use_annotated = use_annotated

        self._buffer = deque(maxlen=int(self.pre_sec * self.fps))
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer: Optional[cv2.VideoWriter] = None
        self._active = False
        self._frames_left = 0
        self._size: Optional[Tuple[int, int]] = None  # (w, h)

    @staticmethod
    def _safe(s: str) -> str:
        s = s or ""
        return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)[:40]

    def _build_path(self, meta: Dict[str, Any]) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cam = self._safe(meta.get("camera_id", ""))
        loc = self._safe(meta.get("location", ""))
        cls_name = self._safe(meta.get("cls_name", "event"))
        fname = f"clip_{ts}_{cam}_{loc}_{cls_name}.mp4"
        return self.out_dir / fname

    def push(self, frame: np.ndarray) -> None:
        self._buffer.append(frame)

    def start(self, meta: Dict[str, Any], frame_size: Tuple[int, int]) -> Optional[Path]:
        if self._active:
            return None

        self._size = frame_size
        out_path = self._build_path(meta)
        self._writer = cv2.VideoWriter(str(out_path), self._fourcc, float(self.fps), self._size)
        if not self._writer or not self._writer.isOpened():
            self._writer = None
            return None

        for f in list(self._buffer):
            self._writer.write(f)

        self._frames_left = int(self.post_sec * self.fps)
        self._active = True
        return out_path

    def tick(self, frame: np.ndarray) -> Optional[bool]:
        if not self._active or self._writer is None:
            return None
        self._writer.write(frame)
        self._frames_left -= 1
        if self._frames_left <= 0:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None
            self._active = False
            return True
        return None


_clip_last_time: Dict[Tuple[str, str, str], float] = {}

# ======================================
# Single-stream legacy generator
# ======================================
def gen_frames(source: str = "webcam", input_uri: Optional[str] = None, save: bool = False, camera_id: str = "", location: str = ""):
    if source == "webcam":
        try:
            cam_index = int(str(input_uri).strip()) if input_uri is not None else 0
        except Exception:
            cam_index = 0
        source_label = f"webcam:{cam_index}"
        cap_source: Union[int, str] = cam_index

    elif source in ("file", "url"):
        if not input_uri:
            yield _encode_mjpeg(_error_image("No file/URL provided."))
            return
        parsed = urlparse(input_uri)
        if source == "file" and not parsed.scheme:
            source_label = Path(input_uri).name
            cap_source = input_uri
        else:
            source_label = input_uri
            cap_source = input_uri
    else:
        yield _encode_mjpeg(_error_image("Unsupported source type"))
        return

    cap = open_capture(cap_source)
    if cap is None:
        app.logger.error("OpenCV could not open source: %s", cap_source)
        yield _encode_mjpeg(_error_image("Failed to open source (camera/file/URL)."))
        return

    csv_logger = CSVLogger(LOG_DIR, source_label, camera_id=camera_id, location=location)

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_idx = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = FPS_FALLBACK

    clip_recorder = EventClipRecorder(out_dir=CLIP_DIR, fps=fps, pre_sec=CLIP_PRE_SEC, post_sec=CLIP_POST_SEC, use_annotated=CLIP_USE_ANNOTATED)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                yield _encode_mjpeg(_error_image("Stream read failed."))
                break

            # --- RT metrics for single-stream preview ---
            capture_ts = time.time()
            t0 = time.perf_counter()
            with _model_lock:
                results = model.predict(frame, verbose=False, conf=PRED_CONF, device=PRED_DEVICE)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            r0 = results[0]
            annotated = r0.plot()

            end2end_ms = (time.time() - capture_ts) * 1000.0
            if OVERLAY_TIMING:
                lag_s = max(0.0, time.time() - capture_ts)
                live = (lag_s <= REALTIME_THRESHOLD_SEC)
                status = "LIVE" if live else f"LAG {lag_s:.1f}s"
                color = (0, 200, 0) if live else (0, 0, 255)
                info = f"{status} | infer {infer_ms:.1f} ms | e2e {end2end_ms:.0f} ms"
                try:
                    cv2.putText(annotated, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                except Exception:
                    pass
            # --------------------------------------------

            if save and writer is None:
                h, w = annotated.shape[:2]
                real_fps = cap.get(cv2.CAP_PROP_FPS)
                if not real_fps or real_fps <= 1:
                    real_fps = FPS_FALLBACK
                out_path = OUTPUT_DIR / f"output_{int(time.time())}.mp4"
                writer = cv2.VideoWriter(str(out_path), fourcc, float(real_fps), (w, h))

            if writer is not None:
                writer.write(annotated)

            clip_frame = annotated if CLIP_USE_ANNOTATED else frame
            clip_recorder.push(clip_frame)
            clip_recorder.tick(clip_frame)

            server_time_iso = datetime.now(timezone.utc).isoformat()
            boxes = r0.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()

                if hasattr(model, "names") and isinstance(model.names, (dict, list)):
                    names = model.names
                elif hasattr(model, "model") and hasattr(model.model, "names"):
                    names = model.model.names
                else:
                    names = {}

                for i in range(len(cls_ids)):
                    conf_val = float(confs[i])
                    if conf_val < MIN_CONF:
                        continue

                    x1, y1, x2, y2 = xyxy[i]
                    cls_id = int(cls_ids[i])

                    if isinstance(names, dict):
                        cls_name = names.get(cls_id, str(cls_id))
                    elif isinstance(names, list) and 0 <= cls_id < len(names):
                        cls_name = names[cls_id]
                    else:
                        cls_name = str(cls_id)

                    if SAFETY_CLASS_WHITELIST and cls_name not in SAFETY_CLASS_WHITELIST:
                        continue

                    csv_logger.log_detection(
                        server_time_iso=server_time_iso,
                        source=source_label,
                        frame_idx=frame_idx,
                        class_id=cls_id,
                        class_name=cls_name,
                        conf=conf_val,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )

                    cls_name_norm = str(cls_name).strip()
                    cls_name_lc = cls_name_norm.lower()

                    # Clip trigger (only for negative if NEGATIVE_CLASSES is set; otherwise all)
                    if CLIP_ENABLED:
                        is_negative_clip = (not alerts_config.negative_classes) or (cls_name_lc in alerts_config.negative_classes)
                        if is_negative_clip:
                            key = (camera_id or "", location or "", cls_name_lc)
                            now_ts = time.time()
                            last_ts = _clip_last_time.get(key, 0.0)
                            if now_ts - last_ts >= CLIP_COOLDOWN_SEC:
                                h, w = annotated.shape[:2]
                                meta = {"camera_id": camera_id or "-", "location": location or "-", "cls_name": cls_name_norm, "source_label": source_label}
                                outp = clip_recorder.start(meta, frame_size=(w, h))
                                if outp:
                                    app.logger.info("Started event clip: %s", outp.name)
                                    _clip_last_time[key] = now_ts

                    # Email alert
                    if alerts_config.enabled:
                        is_negative = (not alerts_config.negative_classes) or (cls_name_lc in alerts_config.negative_classes)
                        if is_negative and alerts_config.recipients and alerts_config.should_alert(camera_id, location, cls_name_norm):
                            context = {
                                "timestamp_utc": server_time_iso,
                                "camera_id": camera_id or "-",
                                "location": location or "-",
                                "source_label": source_label,
                                "cls_name": cls_name_norm,
                                "confidence": f"{conf_val:.2f}",
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "recipients": alerts_config.recipients,
                            }
                            _send_violation_email_async(context, annotated)

            frame_idx += 1
            yield _encode_mjpeg(annotated)

    except GeneratorExit:
        pass
    except Exception as e:
        yield _encode_mjpeg(_error_image(f"Error: {e}"))
    finally:
        try:
            cap.release()
        except Exception:
            pass
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        csv_logger.close()


# ======================================
# Multi-stream infrastructure
# ======================================
class BackgroundSegmentRecorder:
    """
    Continuously writes frames to segment files of fixed duration.
    Thread-safe; call .write(frame) from producer thread.
    """

    def __init__(self, base_dir: Path, stream_id: str, fps: float):
        self.base_dir = base_dir / stream_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.fps = max(float(fps or 0), 1.0)
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer: Optional[cv2.VideoWriter] = None
        self._frames_in_segment = 0
        self._max_frames = int(REC_SEGMENT_SEC * self.fps)
        self._size: Optional[Tuple[int, int]] = None
        self._lock = threading.Lock()

    def _new_path(self) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.base_dir / f"{ts}.mp4"

    def _ensure_writer(self, frame: np.ndarray):
        if self._writer is not None:
            return
        h, w = frame.shape[:2]
        self._size = (w, h)
        out_path = self._new_path()
        self._writer = cv2.VideoWriter(str(out_path), self._fourcc, self.fps, self._size)
        self._frames_in_segment = 0

    def write(self, frame: np.ndarray):
        if not REC_ENABLED:
            return
        with self._lock:
            self._ensure_writer(frame)
            if self._writer is None:
                return
            self._writer.write(frame)
            self._frames_in_segment += 1
            if self._frames_in_segment >= self._max_frames:
                try:
                    self._writer.release()
                except Exception:
                    pass
                self._writer = None
                self._frames_in_segment = 0

    def close(self):
        with self._lock:
            if self._writer is not None:
                try:
                    self._writer.release()
                except Exception:
                    pass
            self._writer = None
            self._frames_in_segment = 0


@dataclass
class MultiStream:
    """
    One live stream running in its own thread:
      - capture -> inference -> annotate
      - CSV logging, event clips, email alerts
      - background continuous recording
      - last MJPEG chunk available for viewers
      - real-time metrics for wall feed health
    """

    stream_id: str
    source: str
    input_uri: Union[str, int]
    camera_id: str = ""
    location: str = ""
    save_background: bool = True

    running: bool = field(default=False, init=False)
    last_chunk: bytes = field(default=b"", init=False)
    last_time: float = field(default=0.0, init=False)
    fps: float = field(default=FPS_FALLBACK, init=False)

    thread: Optional[threading.Thread] = field(default=None, init=False)
    cond: threading.Condition = field(default_factory=threading.Condition, init=False)

    # --- real-time metrics ---
    last_capture_ts: float = field(default=0.0, init=False)   # epoch seconds when the last frame was read
    last_infer_ms: float = field(default=0.0, init=False)     # last inference duration in ms
    last_end2end_ms: float = field(default=0.0, init=False)   # capture->delivered MJPEG chunk in ms
    delivered_fps: float = field(default=0.0, init=False)     # computed from MJPEG notifications
    _last_notify_time: float = field(default=0.0, init=False) # internal for delivered_fps

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, name=f"stream-{self.stream_id}", daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def _notify_chunk(self, chunk: bytes):
        now = time.time()
        with self.cond:
            self.last_chunk = chunk
            self.last_time = now
            # delivered FPS from gaps between notifies
            if self._last_notify_time > 0:
                dt = now - self._last_notify_time
                if dt > 0:
                    self.delivered_fps = 1.0 / dt
            self._last_notify_time = now
            self.cond.notify_all()

    def _run_loop(self):
        cap = None
        writer_csv = None
        clip_recorder = None
        bg_rec = None
        frame_idx = 0

        if self.source == "webcam":
            try:
                cam_index = int(str(self.input_uri).strip())
            except Exception:
                cam_index = 0
            source_label = f"webcam:{cam_index}"
            cap_source: Union[int, str] = cam_index
        else:
            source_label = str(self.input_uri)
            cap_source = self.input_uri

        try:
            cap = open_capture(cap_source)
            if not cap or not cap.isOpened():
                app.logger.error("Failed to open source in stream %s: %s", self.stream_id, cap_source)
                self._notify_chunk(_encode_mjpeg(_error_image("Open failed.")))
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 1:
                fps = FPS_FALLBACK
            self.fps = fps

            writer_csv = CSVLogger(LOG_DIR, source_label, camera_id=self.camera_id, location=self.location)

            clip_recorder = EventClipRecorder(
                out_dir=CLIP_DIR,
                fps=self.fps,
                pre_sec=CLIP_PRE_SEC,
                post_sec=CLIP_POST_SEC,
                use_annotated=CLIP_USE_ANNOTATED,
            )

            if self.save_background:
                bg_rec = BackgroundSegmentRecorder(REC_DIR, self.stream_id, fps=self.fps)

            while self.running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self._notify_chunk(_encode_mjpeg(_error_image("Read failed.")))
                    time.sleep(0.2)
                    continue

                # --- RT metrics ---
                capture_ts = time.time()
                t0 = time.perf_counter()
                with _model_lock:
                    results = model.predict(frame, verbose=False, conf=PRED_CONF, device=PRED_DEVICE)
                infer_ms = (time.perf_counter() - t0) * 1000.0
                r0 = results[0]
                annotated = r0.plot()
                end2end_ms = (time.time() - capture_ts) * 1000.0

                self.last_capture_ts = capture_ts
                self.last_infer_ms = infer_ms
                self.last_end2end_ms = end2end_ms

                if OVERLAY_TIMING:
                    lag_s = max(0.0, time.time() - capture_ts)
                    live = (lag_s <= REALTIME_THRESHOLD_SEC)
                    status = "LIVE" if live else f"LAG {lag_s:.1f}s"
                    color = (0, 200, 0) if live else (0, 0, 255)
                    info = f"{status} | infer {infer_ms:.1f} ms | e2e {end2end_ms:.0f} ms"
                    try:
                        cv2.putText(annotated, info, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    except Exception:
                        pass
                # -------------------

                # background recording
                if bg_rec is not None:
                    try:
                        bg_rec.write(annotated)
                    except Exception:
                        app.logger.exception("BG record failed in %s", self.stream_id)

                # event clip buffer and tick
                clip_frame = annotated if CLIP_USE_ANNOTATED else frame
                clip_recorder.push(clip_frame)
                clip_recorder.tick(clip_frame)

                server_time_iso = datetime.now(timezone.utc).isoformat()
                boxes = r0.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls_ids = boxes.cls.cpu().numpy().astype(int)
                    confs = boxes.conf.cpu().numpy()

                    if hasattr(model, "names") and isinstance(model.names, (dict, list)):
                        names = model.names
                    elif hasattr(model, "model") and hasattr(model.model, "names"):
                        names = model.model.names
                    else:
                        names = {}

                    for i in range(len(cls_ids)):
                        conf_val = float(confs[i])
                        if conf_val < MIN_CONF:
                            continue

                        x1, y1, x2, y2 = xyxy[i]
                        cls_id = int(cls_ids[i])
                        if isinstance(names, dict):
                            cls_name = names.get(cls_id, str(cls_id))
                        elif isinstance(names, list) and 0 <= cls_id < len(names):
                            cls_name = names[cls_id]
                        else:
                            cls_name = str(cls_id)

                        if SAFETY_CLASS_WHITELIST and cls_name not in SAFETY_CLASS_WHITELIST:
                            continue

                        writer_csv.log_detection(
                            server_time_iso=server_time_iso,
                            source=source_label,
                            frame_idx=frame_idx,
                            class_id=cls_id,
                            class_name=cls_name,
                            conf=conf_val,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                        )

                        cls_name_norm = str(cls_name).strip()
                        cls_name_lc = cls_name_norm.lower()

                        # Clip trigger (negative-only if NEGATIVE_CLASSES set)
                        if CLIP_ENABLED:
                            is_negative_clip = (not alerts_config.negative_classes) or (cls_name_lc in alerts_config.negative_classes)
                            if is_negative_clip:
                                key = (self.camera_id or "", self.location or "", cls_name_lc)
                                now_ts = time.time()
                                last_ts = _clip_last_time.get(key, 0.0)
                                if now_ts - last_ts >= CLIP_COOLDOWN_SEC:
                                    h, w = annotated.shape[:2]
                                    meta = {
                                        "camera_id": self.camera_id or "-",
                                        "location": self.location or "-",
                                        "cls_name": cls_name_norm,
                                        "source_label": source_label,
                                    }
                                    outp = clip_recorder.start(meta, frame_size=(w, h))
                                    if outp:
                                        app.logger.info("[%s] Started event clip: %s", self.stream_id, outp.name)
                                        _clip_last_time[key] = now_ts

                        # Email alerts
                        is_negative = (not alerts_config.negative_classes) or (cls_name_lc in alerts_config.negative_classes)
                        if alerts_config.enabled and is_negative and alerts_config.recipients:
                            if alerts_config.should_alert(self.camera_id, self.location, cls_name_norm):
                                context = {
                                    "timestamp_utc": server_time_iso,
                                    "camera_id": self.camera_id or "-",
                                    "location": self.location or "-",
                                    "source_label": source_label,
                                    "cls_name": cls_name_norm,
                                    "confidence": f"{conf_val:.2f}",
                                    "x1": int(x1),
                                    "y1": int(y1),
                                    "x2": int(x2),
                                    "y2": int(y2),
                                    "recipients": alerts_config.recipients,
                                }
                                _send_violation_email_async(context, annotated)

                frame_idx += 1
                self._notify_chunk(_encode_mjpeg(annotated))

        except Exception:
            app.logger.exception("Stream loop crashed: %s", self.stream_id)
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            try:
                if writer_csv is not None:
                    writer_csv.close()
            except Exception:
                pass
            try:
                if bg_rec is not None:
                    bg_rec.close()
            except Exception:
                pass
            self.running = False
            self._notify_chunk(_encode_mjpeg(_error_image("Stopped.")))

    def mjpeg_generator(self):
        while self.running:
            with self.cond:
                timeout = 1.0 / max(self.fps, 1.0)
                self.cond.wait(timeout=timeout)
                chunk = self.last_chunk or _encode_mjpeg(_error_image("Starting..."))
            yield chunk
        yield self.last_chunk or _encode_mjpeg(_error_image("Stopped."))


class MultiStreamManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._streams: Dict[str, MultiStream] = {}

    def list(self) -> List[MultiStream]:
        with self._lock:
            return list(self._streams.values())

    def get(self, stream_id: str) -> Optional[MultiStream]:
        with self._lock:
            return self._streams.get(stream_id)

    def add(
        self,
        source: str,
        input_uri: Union[str, int],
        camera_id: str = "",
        location: str = "",
        stream_id: Optional[str] = None,
        save_background: bool = True,
    ) -> str:
        if not stream_id:
            base = f"{camera_id}_{location}" if (camera_id or location) else str(input_uri)
            sid = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in base)[:40]
            sid = sid or f"cam_{int(time.time())}"
        else:
            sid = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in stream_id)[:40]

        with self._lock:
            if sid in self._streams:
                raise ValueError(f"Stream id already exists: {sid}")
            ms = MultiStream(
                stream_id=sid,
                source=source,
                input_uri=input_uri,
                camera_id=camera_id,
                location=location,
                save_background=save_background,
            )
            self._streams[sid] = ms
            ms.start()
            app.logger.info("Started stream %s (%s, %s)", sid, source, input_uri)
            return sid

    def remove(self, stream_id: str) -> bool:
        with self._lock:
            ms = self._streams.pop(stream_id, None)
        if ms:
            ms.stop()
            app.logger.info("Stopped stream %s", stream_id)
            return True
        return False


streams = MultiStreamManager()

# ======================================
# Auto-start streams if configured
# ======================================
DEFAULT_BOOT = BASE_DIR / "streams_boot.json"
BOOT_PATH = os.getenv("STREAMS_BOOT_JSON", str(DEFAULT_BOOT) if DEFAULT_BOOT.exists() else "")

if BOOT_PATH:
    p = Path(BOOT_PATH)
    logger.info("BOOT_PATH=%s", p)
    if p.exists() and p.is_file():
        try:
            raw = p.read_text(encoding="utf-8", errors="strict").lstrip("\ufeff")
            if raw.strip():
                arr = json.loads(raw)
                if isinstance(arr, list):
                    count = 0
                    for s in arr:
                        try:
                            streams.add(
                                source=(s.get("source") or "url"),
                                input_uri=s.get("input_uri"),
                                camera_id=s.get("camera_id", ""),
                                location=s.get("location", ""),
                                stream_id=s.get("stream_id"),
                                save_background=bool(s.get("save_background", True)),
                            )
                            count += 1
                        except Exception:
                            app.logger.exception("Failed boot start entry: %s", s)
                    app.logger.info("Auto-started %d streams from %s", count, p)
                else:
                    logger.error("Boot JSON must be a list")
            else:
                logger.warning("Boot JSON empty: %s", p)
        except Exception:
            logger.exception("Failed to load boot file: %s", p)

# ======================================
# Routes - Home / Single stream
# ======================================
@app.route("/", methods=["GET"])
def index():
    src = request.args.get("src", "webcam")
    save = request.args.get("save", "0") == "1"
    cam = request.args.get("cam", "0")
    url_input = request.args.get("url", "")
    last_file = request.args.get("file", "")
    camera_id = request.args.get("camera_id", "")
    location = request.args.get("location", "")

    # Build stream URL
    if src == "webcam":
        stream_url = url_for("video_feed", src="webcam", cam=cam, save=int(save),
                             camera_id=camera_id, location=location)
    elif src == "url":
        stream_url = url_for("video_feed", src="url", url=url_input, save=int(save),
                             camera_id=camera_id, location=location)
    else:
        stream_url = url_for("video_feed", src="file", file=last_file, save=int(save),
                             camera_id=camera_id, location=location)

    # NEW: load latest 30 observations
    recent_rows, recent_csv = _get_recent_observations(limit=30)

    return render_template(
        "index.html",
        src=src, save=save,
        cam=cam, url_input=url_input, last_file=last_file,
        stream_url=stream_url,
        camera_id=camera_id, location=location,
        model_path=str(MODEL_PATH),
        upload_dir=str(UPLOAD_DIR),
        output_dir=str(OUTPUT_DIR),
        log_dir=str(LOG_DIR),

        # NEW:
        recent_rows=recent_rows,
        recent_csv=recent_csv,
    )


@app.route("/video_feed")
def video_feed():
    src = request.args.get("src", "webcam")
    save = request.args.get("save", "0") == "1"
    cam = request.args.get("cam", "0")
    url_param = request.args.get("url", "")
    filename = request.args.get("file", "")

    camera_id = request.args.get("camera_id", "")
    location = request.args.get("location", "")

    def _error_stream(msg: str):
        def _gen():
            yield _encode_mjpeg(_error_image(msg))
        return Response(_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    if src == "url":
        if not url_param:
            return _error_stream("No URL provided.")
        return Response(
            gen_frames(source="url", input_uri=url_param, save=save, camera_id=camera_id, location=location),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    if src == "file":
        if not filename:
            return _error_stream("No filename provided.")
        if filename.startswith(("rtsp://", "rtsps://", "http://", "https://")):
            return Response(
                gen_frames(source="url", input_uri=filename, save=save, camera_id=camera_id, location=location),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        base = UPLOAD_DIR.resolve()
        safe_path = (UPLOAD_DIR / filename).resolve()
        if not str(safe_path).startswith(str(base)) or not safe_path.exists():
            return _error_stream("Invalid or missing file.")
        return Response(
            gen_frames(source="file", input_uri=str(safe_path), save=save, camera_id=camera_id, location=location),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    # webcam
    return Response(
        gen_frames(source="webcam", input_uri=cam, save=save, camera_id=camera_id, location=location),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        flash("No file part")
        return redirect(url_for("index", src="file"))

    file = request.files["file"]
    if file.filename == "":
        flash("No selected file")
        return redirect(url_for("index", src="file"))

    if file and allowed_video_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = UPLOAD_DIR / filename
        file.save(save_path)
        flash(f"Uploaded video: {filename}", "success")
        return redirect(url_for("index", src="file", file=filename))

    flash("Unsupported video type", "error")
    return redirect(url_for("index", src="file"))


@app.route("/upload_image", methods=["POST"])
def upload_image():
    camera_id = request.form.get("camera_id", "").strip()
    location = request.form.get("location", "").strip()

    if "image" not in request.files:
        flash("No image part", "error")
        return redirect(url_for("index"))

    image_file = request.files["image"]
    if image_file.filename == "":
        flash("No selected image", "error")
        return redirect(url_for("index"))

    if not allowed_image_file(image_file.filename):
        flash("Unsupported image type", "error")
        return redirect(url_for("index"))

    file_bytes = np.frombuffer(image_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        flash("Failed to decode image", "error")
        return redirect(url_for("index"))

    with _model_lock:
        results = model.predict(img, verbose=False, conf=PRED_CONF, device=PRED_DEVICE)
    r0 = results[0]
    annotated = r0.plot()

    ts = int(time.time())
    out_path = OUTPUT_DIR / f"annotated_{ts}.jpg"
    cv2.imwrite(str(out_path), annotated)

    source_label = f"image:{image_file.filename}"
    csv_logger = CSVLogger(LOG_DIR, source_label, camera_id=camera_id, location=location)
    try:
        server_time_iso = datetime.now(timezone.utc).isoformat()
        boxes = r0.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()

            if hasattr(model, "names") and isinstance(model.names, (dict, list)):
                names = model.names
            elif hasattr(model, "model") and hasattr(model.model, "names"):
                names = model.model.names
            else:
                names = {}

            for i in range(len(cls_ids)):
                conf_val = float(confs[i])
                if conf_val < MIN_CONF:
                    continue
                x1, y1, x2, y2 = xyxy[i]
                cls_id = int(cls_ids[i])
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                if SAFETY_CLASS_WHITELIST and cls_name not in SAFETY_CLASS_WHITELIST:
                    continue
                csv_logger.log_detection(
                    server_time_iso=server_time_iso,
                    source=source_label,
                    frame_idx=0,
                    class_id=cls_id,
                    class_name=cls_name,
                    conf=conf_val,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
    finally:
        csv_logger.close()

    ok, buf = cv2.imencode(".png", annotated)
    data_uri = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    return render_template(
        "image_result.html",
        image_data_uri=data_uri,
        output_filename=out_path.name,
        camera_id=camera_id,
        location=location,
        detections=len(r0.boxes) if r0.boxes is not None else 0,
    )


@app.route("/download-latest")
def download_latest():
    mp4s = sorted(OUTPUT_DIR.glob("output_*.mp4"), key=os.path.getmtime, reverse=True)
    if not mp4s:
        flash("No saved output yet.", "warning")
        return redirect(url_for("index"))
    return send_file(mp4s[0], as_attachment=True, download_name=mp4s[0].name)


@app.route("/download-latest_log")
def download_latest_log():
    csvs = sorted(LOG_DIR.glob("detections_*.csv"), key=os.path.getmtime, reverse=True)
    if not csvs:
        flash("No detection logs available yet.", "warning")
        return redirect(url_for("index"))
    return send_file(csvs[0], as_attachment=True, download_name=csvs[0].name)


def _list_detection_csvs() -> List[Path]:
    return sorted(LOG_DIR.glob("detections_*.csv"), key=os.path.getmtime, reverse=True)

def _latest_detection_csv() -> Optional[Path]:
    """Return newest detection CSV or None."""
    csvs = sorted(LOG_DIR.glob("detections_*.csv"), key=os.path.getmtime, reverse=True)
    return csvs[0] if csvs else None


def _get_recent_observations(limit: int = 30) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Read the newest detection CSV and return the latest N observations as dicts.
    Returns: (rows, csv_name)
    """
    csv_path = _latest_detection_csv()
    if not csv_path:
        return [], None

    try:
        df = pd.read_csv(csv_path)

        required = {"server_time_iso", "camera_id", "location", "class_name", "confidence"}
        if not required.issubset(df.columns):
            for col in required:
                if col not in df.columns:
                    df[col] = ""

        df["ts_utc"] = pd.to_datetime(df["server_time_iso"], utc=True, errors="coerce")
        df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc", ascending=False)

        df = df.head(limit)

        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
        df["confidence"] = df["confidence"].round(2)

        rows = []
        for _, r in df.iterrows():
            rows.append(
                {
                    "time": str(r.get("ts_utc")),
                    "camera_id": str(r.get("camera_id", "") or "-"),
                    "location": str(r.get("location", "") or "-"),
                    "class_name": str(r.get("class_name", "") or "-"),
                    "confidence": float(r.get("confidence", 0.0)),
                }
            )
        return rows, csv_path.name

    except Exception as e:
        app.logger.exception("Failed reading recent observations: %s", e)
        return [], csv_path.name

# ======================================
# Clips
# ======================================
def _list_clips() -> List[Path]:
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(CLIP_DIR.glob("clip_*.mp4"), key=os.path.getmtime, reverse=True)


@app.route("/clips", methods=["GET"])
def clips():
    files = _list_clips()
    rows = []
    for p in files:
        try:
            st = p.stat()
            rows.append(
                {"name": p.name, "size_mb": round(st.st_size / (1024 * 1024), 2), "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")}
            )
        except Exception:
            continue

    return render_template("clips.html", clips=rows, clip_dir=str(CLIP_DIR))


@app.route("/download-clip", methods=["GET"])
def download_clip():
    name = request.args.get("name", "").strip()
    if not name:
        return "Missing ?name=<file>", 400

    base = CLIP_DIR.resolve()
    safe_path = (CLIP_DIR / name).resolve()
    if not str(safe_path).startswith(str(base)) or not safe_path.exists():
        return "Invalid file path or not found", 404

    return send_file(safe_path, as_attachment=True, download_name=safe_path.name)


@app.route("/download-clips-zip", methods=["GET"])
def download_clips_zip():
    files = _list_clips()
    if not files:
        flash("No clips to download.", "warning")
        return redirect(url_for("clips"))

    mem = BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=p.name)
    mem.seek(0)

    fname = f"clips_{int(time.time())}.zip"
    return send_file(mem, as_attachment=True, download_name=fname, mimetype="application/zip")


@app.route("/delete-clip", methods=["POST"])
def delete_clip():
    name = request.form.get("name", "").strip()
    if not name:
        flash("Missing clip name.", "error")
        return redirect(url_for("clips"))

    base = CLIP_DIR.resolve()
    safe_path = (CLIP_DIR / name).resolve()
    if not str(safe_path).startswith(str(base)) or not safe_path.exists():
        flash("Invalid clip path or not found.", "error")
        return redirect(url_for("clips"))

    try:
        os.remove(safe_path)
        flash(f"Deleted {safe_path.name}.", "success")
    except Exception as e:
        app.logger.exception("Failed to delete clip %s: %s", safe_path, e)
        flash("Failed to delete clip; see server logs.", "error")
    return redirect(url_for("clips"))


# ======================================
# Analytics (unchanged logic, just ensure plotly js in base.html)
# ======================================
def _load_and_filter_csv_for_analytics(args) -> Dict[str, Any]:
    csv_options = [p.name for p in _list_detection_csvs()]

    file_param = args.get("file", "").strip()
    stacked = args.get("stacked", "1") == "1"
    top_k = int(args.get("top_k", "15"))

    if file_param:
        base = LOG_DIR.resolve()
        safe_path = (LOG_DIR / file_param).resolve()
        if not str(safe_path).startswith(str(base)) or not safe_path.exists():
            raise FileNotFoundError(f"File not found: {file_param}")
        csv_path = safe_path
    else:
        csvs = _list_detection_csvs()
        if not csvs:
            raise FileNotFoundError("No detection logs available yet. Stream something first.")
        csv_path = csvs[0]

    df = pd.read_csv(csv_path)

    required_cols = {"server_time_iso", "class_name", "confidence"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if "camera_id" not in df.columns:
        df["camera_id"] = ""
    if "location" not in df.columns:
        df["location"] = ""

    df["ts_utc"] = pd.to_datetime(df["server_time_iso"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"])

    try:
        min_conf = float(args.get("min_conf", str(MIN_CONF)))
    except Exception:
        min_conf = MIN_CONF
    df = df[df["confidence"] >= min_conf]

    try:
        tz_offset_min = int(args.get("tz_offset", "0"))
    except Exception:
        tz_offset_min = 0
    df["ts_local"] = df["ts_utc"] + pd.to_timedelta(tz_offset_min, unit="m")

    start_str = args.get("start", "").strip()
    end_str = args.get("end", "").strip()
    if start_str:
        try:
            start_dt = pd.to_datetime(start_str).tz_localize(None)
            df = df[df["ts_local"] >= start_dt]
        except Exception:
            pass
    if end_str:
        try:
            end_dt = pd.to_datetime(end_str)
            if end_dt.time() == pd.Timestamp.min.time():
                end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            end_dt = end_dt.tz_localize(None)
            df = df[df["ts_local"] <= end_dt]
        except Exception:
            pass

    cam_filter = args.get("camera_id", "").strip()
    loc_filter = args.get("location", "").strip()
    if cam_filter:
        df = df[df["camera_id"].astype(str).str.contains(cam_filter, case=False, na=False)]
    if loc_filter:
        df = df[df["location"].astype(str).str.contains(loc_filter, case=False, na=False)]

    def parse_list(s: str) -> List[str]:
        return [x.strip() for x in s.split(",") if x.strip()]

    class_in = parse_list(args.get("class_in", ""))
    class_ex = parse_list(args.get("class_ex", ""))
    if class_in:
        df = df[df["class_name"].astype(str).isin(class_in)]
    if class_ex:
        df = df[~df["class_name"].astype(str).isin(class_ex)]

    if df.empty:
        return {
            "csv_path": csv_path,
            "csv_options": csv_options,
            "df": df,
            "hourly_counts": pd.DataFrame(columns=["hour", "count"]),
            "daily_counts": pd.DataFrame(columns=["date", "count"]),
            "class_counts": pd.DataFrame(columns=["class_name", "count"]),
            "daily_cat": pd.DataFrame(),
            "heatmap_hour_class": pd.DataFrame(),
            "tz_offset_min": tz_offset_min,
            "top_k": top_k,
            "min_conf": min_conf,
            "stacked": stacked,
            "camera_id": cam_filter,
            "location": loc_filter,
            "class_in": class_in,
            "class_ex": class_ex,
            "file_param": file_param,
        }

    df["hour"] = df["ts_local"].dt.floor(pd.Timedelta(hours=1))
    hourly_counts = df.groupby("hour").size().reset_index(name="count")

    df["date"] = df["ts_local"].dt.date
    daily_counts = df.groupby("date").size().reset_index(name="count")

    class_counts = df["class_name"].value_counts().head(top_k).reset_index()
    class_counts.columns = ["class_name", "count"]

    daily_cat = df.groupby(["date", "class_name"]).size().unstack(fill_value=0)

    if stacked and daily_cat.shape[1] > top_k:
        top_classes = class_counts["class_name"].tolist()
        keep = [c for c in daily_cat.columns if c in top_classes]
        other_cols = [c for c in daily_cat.columns if c not in keep]
        if other_cols:
            daily_cat["Other"] = daily_cat[other_cols].sum(axis=1)
        daily_cat = daily_cat[keep + (["Other"] if "Other" in daily_cat.columns else [])]

    top_classes = class_counts["class_name"].tolist()
    df_top = df[df["class_name"].isin(top_classes)].copy()
    heatmap_hour_class = df_top.groupby([df_top["hour"], df_top["class_name"]]).size().unstack(fill_value=0).T

    return {
        "csv_path": csv_path,
        "csv_options": csv_options,
        "df": df,
        "hourly_counts": hourly_counts,
        "daily_counts": daily_counts,
        "class_counts": class_counts,
        "daily_cat": daily_cat,
        "heatmap_hour_class": heatmap_hour_class,
        "tz_offset_min": tz_offset_min,
        "top_k": top_k,
        "min_conf": min_conf,
        "stacked": stacked,
        "camera_id": cam_filter,
        "location": loc_filter,
        "class_in": class_in,
        "class_ex": class_ex,
        "file_param": file_param,
    }


def plot_hourly(hourly_df: pd.DataFrame, tz_offset: int) -> str:
    fig = px.bar(hourly_df, x="hour", y="count", title=f"Observations per Hour (tz_offset {tz_offset} min)", labels={"hour": "Hour", "count": "Count"})
    fig.update_layout(hovermode="x unified", xaxis_tickformat="%Y-%m-%d %H:%M", margin=dict(l=40, r=20, t=60, b=40))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plot_daily(daily_df: pd.DataFrame) -> str:
    fig = px.bar(daily_df, x="date", y="count", title="Observations per Day", labels={"date": "Date", "count": "Count"})
    fig.update_layout(hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plot_trend_line(daily_df: pd.DataFrame) -> str:
    fig = px.line(daily_df, x="date", y="count", markers=True, title="Observation Trend Over Time", labels={"date": "Date", "count": "Count"})
    fig.update_layout(hovermode="x unified", margin=dict(l=40, r=20, t=60, b=40))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plot_top_classes(class_df: pd.DataFrame, top_k: int) -> str:
    fig = px.bar(class_df, x="class_name", y="count", title=f"Top {top_k} Classes", labels={"class_name": "Class", "count": "Count"})
    fig.update_layout(xaxis={"categoryorder": "total descending"}, margin=dict(l=40, r=20, t=60, b=80))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plot_class_pie(class_df: pd.DataFrame) -> str:
    if class_df.empty:
        return ""
    fig = px.pie(class_df, names="class_name", values="count", title="Class Distribution")
    fig.update_traces(textposition="inside", textinfo="percent+label", pull=[0.02] * len(class_df))
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=20))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plot_daily_by_class(daily_cat_df: pd.DataFrame, stacked: bool) -> str:
    if daily_cat_df.empty:
        return ""
    df_long = daily_cat_df.copy().reset_index().rename(columns={"index": "date"})
    df_long = df_long.melt(id_vars=["date"], var_name="class_name", value_name="count")
    fig = px.bar(
        df_long,
        x="date",
        y="count",
        color="class_name",
        title=f"Daily Counts by Class{' (stacked)' if stacked else ''}",
        labels={"date": "Date", "count": "Count", "class_name": "Class"},
    )
    fig.update_layout(barmode="stack" if stacked else "group", margin=dict(l=40, r=20, t=60, b=60), legend_traceorder="normal")
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


def plot_heatmap(heatmap_df: pd.DataFrame) -> str:
    if heatmap_df.empty:
        return ""
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_df.values,
            x=[str(c) for c in heatmap_df.columns],
            y=[str(i) for i in heatmap_df.index],
            colorscale="YlOrRd",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(title="Class vs Hour Heatmap", xaxis_title="Hour", yaxis_title="Class", margin=dict(l=60, r=20, t=60, b=80))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


@app.route("/analytics", methods=["GET"])
def analytics():
    try:
        ctx = _load_and_filter_csv_for_analytics(request.args)
    except FileNotFoundError as e:
        flash(str(e), "warning")
        return redirect(url_for("index"))
    except ValueError as e:
        return str(e), 400
    except Exception as e:
        return f"Failed to read CSV: {e}", 500

    df = ctx["df"]
    if df.empty:
        flash("No data after filters. Try relaxing filters.", "warning")
        return render_template(
            "analytics.html",
            csv_name=Path(ctx["csv_path"]).name,
            figs={},
            tz_offset_min=ctx["tz_offset_min"],
            top_k=ctx["top_k"],
            stacked=ctx["stacked"],
            min_conf=ctx["min_conf"],
            camera_id=ctx["camera_id"],
            location=ctx["location"],
            class_in=",".join(ctx["class_in"]),
            class_ex=",".join(ctx["class_ex"]),
            file_param=ctx["file_param"],
            csv_options=ctx["csv_options"],
            empty=True,
        )

    figs: Dict[str, str] = {}
    figs["hourly"] = plot_hourly(ctx["hourly_counts"], ctx["tz_offset_min"])
    figs["daily"] = plot_daily(ctx["daily_counts"])
    figs["trend_line"] = plot_trend_line(ctx["daily_counts"])
    figs["top_classes"] = plot_top_classes(ctx["class_counts"], ctx["top_k"])
    figs["class_pie"] = plot_class_pie(ctx["class_counts"])
    figs["daily_by_class"] = plot_daily_by_class(ctx["daily_cat"], ctx["stacked"])
    if not ctx["heatmap_hour_class"].empty:
        figs["heatmap"] = plot_heatmap(ctx["heatmap_hour_class"])

    return render_template(
        "analytics.html",
        csv_name=Path(ctx["csv_path"]).name,
        figs=figs,
        tz_offset_min=ctx["tz_offset_min"],
        top_k=ctx["top_k"],
        stacked=ctx["stacked"],
        min_conf=ctx["min_conf"],
        camera_id=ctx["camera_id"],
        location=ctx["location"],
        class_in=",".join(ctx["class_in"]),
        class_ex=",".join(ctx["class_ex"]),
        file_param=ctx["file_param"],
        csv_options=ctx["csv_options"],
        empty=False,
    )


@app.route("/analytics_json", methods=["GET"])
def analytics_json():
    try:
        ctx = _load_and_filter_csv_for_analytics(request.args)
    except FileNotFoundError as e:
        return jsonify({"status": "ok", "message": str(e), "data": {}}), 200
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to read CSV: {e}"}), 500

    df = ctx["df"]
    hourly = [{"hour": str(r["hour"]), "count": int(r["count"])} for _, r in ctx["hourly_counts"].iterrows()]
    daily = [{"date": str(r["date"]), "count": int(r["count"])} for _, r in ctx["daily_counts"].iterrows()]
    top_classes = [{"class_name": str(r["class_name"]), "count": int(r["count"])} for _, r in ctx["class_counts"].iterrows()]

    heatmap = []
    heatmap_index = []
    heatmap_columns = []
    if not ctx["heatmap_hour_class"].empty:
        hm = ctx["heatmap_hour_class"]
        heatmap = hm.values.tolist()
        heatmap_index = [str(i) for i in hm.index]
        heatmap_columns = [str(c) for c in hm.columns]

    return jsonify(
        {
            "status": "ok",
            "csv": Path(ctx["csv_path"]).name,
            "filters": {
                "min_conf": ctx["min_conf"],
                "tz_offset_min": ctx["tz_offset_min"],
                "top_k": ctx["top_k"],
                "stacked": ctx["stacked"],
                "start": request.args.get("start", "").strip(),
                "end": request.args.get("end", "").strip(),
                "camera_id": ctx["camera_id"],
                "location": ctx["location"],
                "class_in": ctx["class_in"],
                "class_ex": ctx["class_ex"],
                "file": ctx["file_param"],
            },
            "summary": {"total_rows_after_filters": int(df.shape[0]) if not df.empty else 0},
            "hourly_counts": hourly,
            "daily_counts": daily,
            "top_classes": top_classes,
            "heatmap": {"matrix": heatmap, "rows_class": heatmap_index, "cols_hour": heatmap_columns},
        }
    ), 200


@app.route('/api/obs/summary/today')
def obs_summary_today():
    # Adjust to your actual CSV/log directory & fieldnames
    log_dir = Path("logs")  # or wherever your csv lives
    today_utc = datetime.now(timezone.utc).date()

    class_counts = Counter()
    buckets = defaultdict(int)  # key = "HH:MM" in UTC (30-min steps)

    # Read all relevant CSVs and aggregate
    for csv_path in log_dir.glob("*.csv"):
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Expect fields: server_time_iso, class_name, etc.
                try:
                    t = datetime.fromisoformat(row['server_time_iso'].replace('Z', '+00:00'))
                except Exception:
                    continue
                if t.date() != today_utc:
                    continue
                cls = row.get('class_name', 'Unknown') or 'Unknown'
                class_counts[cls] += 1

                # bucket: every 30 min
                minute = (t.minute // 30) * 30
                key = f"{t.hour:02d}:{minute:02d}"
                buckets[key] += 1

    # Fill missing buckets so the chart looks continuous from 00:00 to now
    now_utc = datetime.now(timezone.utc)
    labels = []
    counts = []
    for h in range(0, now_utc.hour + 1):
        for m in (0, 30):
            if h == now_utc.hour and m > (0 if now_utc.minute < 30 else 30):
                break
            lab = f"{h:02d}:{m:02d}"
            labels.append(lab)
            counts.append(buckets.get(lab, 0))

    return jsonify({
      "classLabels": list(class_counts.keys()),
      "classData": [class_counts[k] for k in class_counts.keys()],
      "timeLabels": labels,
      "timeData": counts
    })


@app.route("/analytics_pdf", methods=["GET"])
def analytics_pdf():
    try:
        ctx = _load_and_filter_csv_for_analytics(request.args)
    except Exception as e:
        return str(e), 200

    df = ctx["df"]
    if df.empty:
        return "No data after filters to export.", 200

    figs = []

    fig1, ax1 = plt.subplots(figsize=(9, 3))
    hc = ctx["hourly_counts"]
    ax1.bar(hc["hour"].astype(str), hc["count"], color="#2E86DE")
    ax1.set_title(f"Observations per Hour (tz_offset {ctx['tz_offset_min']} min)")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", labelrotation=45)
    buf1 = BytesIO()
    fig1.tight_layout()
    fig1.savefig(buf1, format="png", dpi=140)
    plt.close(fig1)
    buf1.seek(0)
    figs.append(buf1)

    fig2, ax2 = plt.subplots(figsize=(9, 3))
    dc = ctx["daily_counts"]
    ax2.bar(dc["date"].astype(str), dc["count"], color="#1ABC9C")
    ax2.set_title("Observations per Day")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", labelrotation=45)
    buf2 = BytesIO()
    fig2.tight_layout()
    fig2.savefig(buf2, format="png", dpi=140)
    plt.close(fig2)
    buf2.seek(0)
    figs.append(buf2)

    fig3, ax3 = plt.subplots(figsize=(9, 3))
    tc = ctx["class_counts"]
    ax3.bar(tc["class_name"].astype(str), tc["count"], color="#9B59B6")
    ax3.set_title(f"Top {ctx['top_k']} Classes")
    ax3.set_xlabel("Class")
    ax3.set_ylabel("Count")
    ax3.tick_params(axis="x", labelrotation=45)
    buf3 = BytesIO()
    fig3.tight_layout()
    fig3.savefig(buf3, format="png", dpi=140)
    plt.close(fig3)
    buf3.seek(0)
    figs.append(buf3)

    fig4, ax4 = plt.subplots(figsize=(9.5, 3.5))
    ctx["daily_cat"].plot(kind="bar", stacked=ctx["stacked"], ax=ax4, colormap="tab20")
    ax4.set_title("Daily Counts by Class" + (" (stacked)" if ctx["stacked"] else ""))
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Count")
    ax4.legend(loc="upper right", fontsize="small", ncol=2)
    buf4 = BytesIO()
    fig4.tight_layout()
    fig4.savefig(buf4, format="png", dpi=140)
    plt.close(fig4)
    buf4.seek(0)
    figs.append(buf4)

    heatmap = ctx["heatmap_hour_class"]
    if not heatmap.empty:
        fig5, ax5 = plt.subplots(figsize=(9.5, 4))
        im = ax5.imshow(heatmap.values, aspect="auto", cmap="YlOrRd")
        ax5.set_title("Class vs Hour Heatmap")
        ax5.set_xlabel("Hour")
        ax5.set_ylabel("Class")
        ax5.set_yticks(range(len(heatmap.index)))
        ax5.set_yticklabels([str(i) for i in heatmap.index])
        ax5.set_xticks(range(len(heatmap.columns)))
        ax5.set_xticklabels([str(c) for c in heatmap.columns], rotation=45, ha="right")
        plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
        buf5 = BytesIO()
        fig5.tight_layout()
        fig5.savefig(buf5, format="png", dpi=140)
        plt.close(fig5)
        buf5.seek(0)
        figs.append(buf5)

    pdf_buf = BytesIO()
    c = canvas.Canvas(pdf_buf, pagesize=landscape(A4))
    width, height = landscape(A4)

    margin = 20
    y = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, f"Safety Observations Analytics: {Path(ctx['csv_path']).name}")
    y -= 18
    c.setFont("Helvetica", 10)
    meta = f"min_conf={ctx['min_conf']}  tz_offset_min={ctx['tz_offset_min']}  top_k={ctx['top_k']}  stacked={'yes' if ctx['stacked'] else 'no'}"
    c.drawString(margin, y, meta)
    y -= 14

    img_w = width - 2 * margin
    for fbuf in figs:
        img = ImageReader(fbuf)
        iw, ih = img.getSize()
        scale = img_w / iw
        draw_h = ih * scale
        if y - draw_h < margin:
            c.showPage()
            y = height - margin
        c.drawImage(img, margin, y - draw_h, width=img_w, height=draw_h)
        y -= draw_h + 10

    c.showPage()
    c.save()
    pdf_buf.seek(0)
    fname = f"analytics_{int(time.time())}.pdf"
    return send_file(pdf_buf, as_attachment=True, download_name=fname, mimetype="application/pdf")


# ======================================
# Alerts config UI
# ======================================
@app.route("/alerts", methods=["GET", "POST"])
def alerts():
    if request.method == "POST":
        enabled = "enabled" in request.form

        recipients_raw = request.form.get("recipients", "")
        recipients = [e.strip() for e in recipients_raw.replace("\n", ",").split(",") if e.strip()]

        cooldown_sec = int(request.form.get("cooldown_sec", alerts_config.cooldown_sec))

        negative_raw = request.form.get("negative_classes", "")
        negative_classes = {s.strip() for s in negative_raw.replace("\n", ",").split(",") if s.strip()}

        alerts_config.update(enabled=enabled, recipients=recipients, cooldown_sec=cooldown_sec, negative_classes=negative_classes)
        flash("Alert settings updated.", "success")
        return redirect(url_for("alerts"))

    cfg = {
        "enabled": alerts_config.enabled,
        "recipients": ", ".join(alerts_config.recipients),
        "cooldown_sec": alerts_config.cooldown_sec,
        "negative_classes": ", ".join(sorted(alerts_config.negative_classes)),
        "smtp_host": email_alerter.smtp_host,
        "smtp_port": email_alerter.smtp_port,
        "from_email": email_alerter.from_email,
        "use_tls": email_alerter.use_tls,
        "has_auth": bool(email_alerter.smtp_user),
    }
    return render_template("alerts.html", cfg=cfg)


@app.route("/send-test-email")
def send_test_email():
    recipients_raw = request.args.get("to", os.getenv("ALERT_RECIPIENTS", ""))
    recipients = [e.strip() for e in recipients_raw.split(",") if e.strip()]
    if not recipients:
        return "No recipients configured. Set ALERT_RECIPIENTS or pass ?to=a@b.com", 400

    subject = "[HSE Alert] Test email"
    html_body = """
    <html><body>
        <h3>Test email from HSE Alerts</h3>
        <p>If you can read this, SMTP is working.</p>
    </body></html>
    """.strip()

    ok = email_alerter.send(subject, html_body, recipients, attachments=None)
    return ("OK: sent" if ok else "FAILED: see server logs"), (200 if ok else 500)


# ======================================
# Multi-stream routes (FIXED + metrics)
# ======================================
@app.route("/streams", methods=["GET", "POST"])
def streams_endpoint():
    """
    GET: redirect to /wall
    POST JSON: {source: 'webcam'|'url'|'file', input_uri, camera_id, location, stream_id?, save_background?}
    """
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        source = (data.get("source") or "").strip().lower()
        input_uri = data.get("input_uri")
        camera_id = (data.get("camera_id") or "").strip()
        location = (data.get("location") or "").strip()
        stream_id = (data.get("stream_id") or "").strip() or None
        save_bg = bool(data.get("save_background", True))

        if source not in ("webcam", "url", "file"):
            return jsonify({"status": "error", "message": "source must be webcam|url|file"}), 400
        if input_uri is None or input_uri == "":
            return jsonify({"status": "error", "message": "input_uri required"}), 400
        if source == "webcam":
            try:
                input_uri = int(str(input_uri).strip())
            except Exception:
                return jsonify({"status": "error", "message": "input_uri must be webcam index int"}), 400

        try:
            sid = streams.add(source=source, input_uri=input_uri, camera_id=camera_id, location=location, stream_id=stream_id, save_background=save_bg)
            return jsonify({"status": "ok", "id": sid}), 200
        except Exception as e:
            app.logger.exception("Failed to add stream")
            return jsonify({"status": "error", "message": str(e)}), 500

    return redirect(url_for("wall"))


@app.route("/streams/json", methods=["GET"])
def streams_json():
    now = time.time()
    payload = []
    for s in streams.list():
        age_ms = None
        is_live = False
        last_ts_utc = None

        if s.last_capture_ts and s.last_capture_ts > 0:
            age_ms = round((now - s.last_capture_ts) * 1000.0, 1)
            is_live = ((now - s.last_capture_ts) <= REALTIME_THRESHOLD_SEC)
            try:
                last_ts_utc = datetime.utcfromtimestamp(s.last_capture_ts).replace(tzinfo=timezone.utc).isoformat()
            except Exception:
                last_ts_utc = None

        payload.append(
            {
                "id": s.stream_id,
                "source": s.source,
                "input_uri": s.input_uri if isinstance(s.input_uri, int) else str(s.input_uri),
                "camera_id": s.camera_id,
                "location": s.location,
                "running": s.running,

                "reported_fps": round(float(s.fps or 0.0), 2),
                "delivered_fps": round(float(s.delivered_fps or 0.0), 2),
                "last_frame_ts_utc": last_ts_utc,
                "age_ms": age_ms,
                "infer_ms": round(float(s.last_infer_ms), 1) if s.last_infer_ms else None,
                "end_to_end_ms": round(float(s.last_end2end_ms), 1) if s.last_end2end_ms else None,
                "is_realtime": bool(is_live),
                "recording": s.save_background and REC_ENABLED,
                "realtime_threshold_sec": REALTIME_THRESHOLD_SEC,
            }
        )
    return jsonify({"status": "ok", "streams": payload})


@app.route("/streams/<stream_id>", methods=["DELETE"])
def delete_stream(stream_id: str):
    ok = streams.remove(stream_id)
    return (jsonify({"status": "ok"}) if ok else jsonify({"status": "not_found"})), (200 if ok else 404)


@app.route("/stream/<stream_id>.mjpeg")
def stream_mjpeg(stream_id: str):
    ms = streams.get(stream_id)
    if not ms:
        def _gen():
            yield _encode_mjpeg(_error_image("Unknown stream id"))
        return Response(_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")
    return Response(ms.mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/view/<stream_id>")
def view_stream(stream_id: str):
    ms = streams.get(stream_id)
    if not ms:
        return f"Stream {stream_id} not found.", 404
    return render_template("view_stream.html", stream=ms)


@app.route("/wall")
def wall():
    return render_template("wall.html", streams=streams.list(), rec_enabled=REC_ENABLED)


@app.route("/healthz")
def healthz():
    return "ok", 200


@app.route("/streamlit")
def streamlit():
    return """
    <iframe src="http://localhost:8501" width="100%" height="900px" style="border:none;">
    </iframe>
    """, 200

if __name__ == "__main__":
    # Auto-start Streamlit
    import subprocess
    subprocess.Popen(["streamlit", "run", "streamlit_app.py", "--server.port=8501"])

    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug, threaded=True)