"""Anime-character eye/face detection wrapper around YOLOv8."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

from PIL import Image

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Known auto-download sources for face detection models (Bingsu/adetailer)
_DOWNLOAD_URLS: dict[str, str] = {
    "face_yolov8n.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt",
    "face_yolov8s.pt": "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt",
}


@dataclass
class EyeBox:
    x: int
    y: int
    w: int
    h: int
    confidence: float

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


class EyeDetector:
    """Wraps an ultralytics YOLO model.

    The bundled default model is a face detector (``face_yolov8n.pt``).
    The face box is split into two eye regions in the upper half of the
    face as a heuristic when the model returns only face boxes.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        confidence_threshold: float = 0.35,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self._model = None
        if not self.model_path.exists():
            logger.warning(
                "Detection model missing: %s. Place a model in models/detection/.",
                self.model_path,
            )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        if not self.model_path.exists():
            self._auto_download()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Detection model not found: {self.model_path}. "
                "Place a YOLO face model (e.g. face_yolov8n.pt) in models/detection/."
            )
        self._model = YOLO(str(self.model_path))

    def _auto_download(self) -> None:
        filename = self.model_path.name
        url = _DOWNLOAD_URLS.get(filename)
        if not url:
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading detection model: %s -> %s", url, self.model_path)
        try:
            urlretrieve(url, str(self.model_path))
            logger.info("Detection model downloaded successfully")
        except Exception as exc:
            logger.error("Auto-download failed (%s). Please place the file manually.", exc)

    def detect(self, image: Image.Image) -> list[EyeBox]:
        self._ensure_loaded()
        results = self._model.predict(
            image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        boxes_out: list[EyeBox] = []
        result = results[0]
        names = getattr(result, "names", {}) or {}
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_idx = int(box.cls[0].item()) if box.cls is not None else -1
            cls_name = names.get(cls_idx, "") if isinstance(names, dict) else ""

            x1, y1, x2, y2 = map(int, xyxy)
            w = x2 - x1
            h = y2 - y1

            if "eye" in str(cls_name).lower():
                boxes_out.append(EyeBox(x=x1, y=y1, w=w, h=h, confidence=conf))
            else:
                # treat as face -> split into two eye regions (heuristic)
                eye_h = int(h * 0.18)
                eye_w = int(w * 0.30)
                y_eye = y1 + int(h * 0.38)
                left_x = x1 + int(w * 0.12)
                right_x = x2 - int(w * 0.12) - eye_w
                boxes_out.append(
                    EyeBox(x=left_x, y=y_eye, w=eye_w, h=eye_h, confidence=conf)
                )
                boxes_out.append(
                    EyeBox(x=right_x, y=y_eye, w=eye_w, h=eye_h, confidence=conf)
                )

        boxes_out.sort(key=lambda b: b.confidence, reverse=True)
        return boxes_out
