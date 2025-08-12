import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlretrieve

import torch
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

try:
    import tensorflow as tf  # type: ignore
    _TF_AVAILABLE = True
except Exception:  # pragma: no cover
    _TF_AVAILABLE = False


class YoloDetector:
    def __init__(self, weights_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    @torch.inference_mode()
    def detect(self, image: Image.Image):
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        return results


class TimmClassifier:
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        class_names: List[str],
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.model.eval()
        self.model.to(self.device)

        self.config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**self.config)

        if weights_path and os.path.exists(weights_path):
            ext = os.path.splitext(weights_path)[1].lower()
            # Only try to load PyTorch checkpoints. Skip .keras/.h5 files which
            # require TensorFlow (handled by KerasClassifier when available).
            if ext in {".pt", ".pth", ".bin"}:
                state = torch.load(weights_path, map_location="cpu")
                # Handle checkpoints that wrap weights in dicts
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.model.load_state_dict(state, strict=False)

        self.class_names = class_names

    @torch.inference_mode()
    def predict(self, image: Image.Image, topk: int = 3) -> Tuple[List[str], List[float]]:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        k = min(topk, probs.shape[0])
        top_probs, top_idxs = torch.topk(probs, k)
        labels = [self.class_names[idx] if idx < len(self.class_names) else str(idx) for idx in top_idxs.tolist()]
        return labels, top_probs.tolist()


class KerasClassifier:
    def __init__(
        self,
        weights_path: str,
        class_names: List[str],
        model_name: str,
        num_classes: int,
    ) -> None:
        if not _TF_AVAILABLE:
            raise RuntimeError("TensorFlow is not installed but a .keras model was configured.")
        self.class_names = class_names
        self.model = None
        # Default sizes per common backbones
        self.input_size = (224, 224)
        lname = model_name.lower()
        if "efficientnet_b0" in lname or "mobilenet_v2" in lname:
            self.input_size = (224, 224)

        # Try to load the full model first
        try:
            self.model = tf.keras.models.load_model(weights_path, safe_mode=False)
            # Infer input size from the loaded model
            shape = self.model.input_shape
            if shape and isinstance(shape, (list, tuple)):
                h, w = shape[1], shape[2]
                if isinstance(h, int) and isinstance(w, int):
                    self.input_size = (w, h)
            return
        except Exception:
            pass

        # Fallback: rebuild a compatible model that accepts 3-channel RGB input directly
        inp = tf.keras.Input(shape=(self.input_size[1], self.input_size[0], 3))
        x = inp
        if "efficientnet" in lname:
            base = tf.keras.applications.efficientnet.EfficientNetB0(
                include_top=True, weights=None, classes=num_classes
            )
        elif "mobilenet" in lname and "v2" in lname:
            base = tf.keras.applications.mobilenet_v2.MobileNetV2(
                include_top=True, weights=None, classes=num_classes
            )
        elif "mobilenet" in lname:
            base = tf.keras.applications.mobilenet.MobileNet(
                include_top=True, weights=None, classes=num_classes
            )
        else:
            base = tf.keras.applications.efficientnet.EfficientNetB0(
                include_top=True, weights=None, classes=num_classes
            )
        out = base(x)
        model = tf.keras.Model(inp, out)
        try:
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        except Exception:
            pass
        self.model = model

    def _preprocess(self, image: Image.Image):
        # Determine expected input dimensions
        if hasattr(self.model, "input_shape") and isinstance(self.model.input_shape, (list, tuple)):
            h, w = self.model.input_shape[1], self.model.input_shape[2]
            if isinstance(h, int) and isinstance(w, int):
                self.input_size = (w, h)
        # Always feed RGB to fallback/loaded models
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img = image.resize(self.input_size)
        arr = tf.keras.preprocessing.image.img_to_array(img)
        arr = arr / 255.0
        arr = tf.expand_dims(arr, axis=0)
        return arr

    def predict(self, image: Image.Image, topk: int = 3) -> Tuple[List[str], List[float]]:
        arr = self._preprocess(image)
        preds = self.model.predict(arr, verbose=0)[0]
        k = min(topk, preds.shape[0])
        top_idxs = preds.argsort()[-k:][::-1]
        labels = [self.class_names[idx] if idx < len(self.class_names) else str(idx) for idx in top_idxs]
        probs = [float(preds[i]) for i in top_idxs]
        return labels, probs


def build_classifier_from_config(cfg: dict) -> Optional[object]:
    model_name = str(cfg.get("model_name", "efficientnet_b0"))
    weights_path = cfg.get("weights_path")
    weights_url: Optional[str] = cfg.get("weights_url")
    class_names: List[str] = cfg.get("class_names", [])
    num_classes = int(cfg.get("num_classes", len(class_names) or 1000))

    # Normalize some common aliases to valid timm model names
    def _normalize_timm_name(name: str) -> str:
        lname = name.lower().strip()
        alias_map = {
            "mobilenet_v2": "mobilenetv2_100",
            "mobilenetv2": "mobilenetv2_100",
            "mobilenet_v3_large": "mobilenetv3_large_100",
            "mobilenet_v3_small": "mobilenetv3_small_100",
            "efficientnet-b0": "efficientnet_b0",
        }
        return alias_map.get(lname, lname)

    model_name = _normalize_timm_name(model_name)

    # Optionally download classifier weights if a URL is provided and the file is missing
    def _is_lfs_pointer(p: Path) -> bool:
        try:
            if not p.exists() or p.stat().st_size > 1024 * 50:
                return False
            head = p.read_text(errors="ignore")
            return "git-lfs.github.com" in head
        except Exception:
            return False

    if isinstance(weights_path, str) and weights_path:
        dest = Path(weights_path)
        if (
            isinstance(weights_url, str)
            and weights_url.startswith(("http://", "https://"))
            and (not dest.exists() or _is_lfs_pointer(dest))
        ):
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                urlretrieve(weights_url, dest.as_posix())
            except Exception:
                # Continue without downloaded weights; downstream code will handle absence
                pass

    # Prefer TensorFlow/Keras weights only if TensorFlow is available
    if (
        _TF_AVAILABLE
        and weights_path
        and isinstance(weights_path, str)
        and os.path.exists(weights_path)
    ):
        ext = os.path.splitext(weights_path)[1].lower()
        if ext in {".keras", ".h5"}:
            try:
                return KerasClassifier(
                    weights_path=weights_path,
                    class_names=class_names,
                    model_name=model_name,
                    num_classes=num_classes,
                )
            except Exception:
                # Fall back to timm below if Keras model load fails
                pass

    # Fall back to timm/PyTorch
    return TimmClassifier(
        model_name=model_name,
        num_classes=num_classes,
        class_names=class_names if class_names else [str(i) for i in range(num_classes)],
        weights_path=weights_path,
    )
