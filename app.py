import os
from pathlib import Path
from urllib.request import urlretrieve
from typing import List, Optional, Tuple

import streamlit as st
from PIL import Image
import yaml

from models.model_loader import YoloDetector, TimmClassifier, build_classifier_from_config
from utils.image_ops import crop_xyxy


st.set_page_config(page_title="Deteksi Dokumen ID + Klasifikasi", layout="wide")


@st.cache_resource(show_spinner=False)
def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_best_id_box(yolo_result, target_class_name: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    boxes = yolo_result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    # Coba cocokkan berdasarkan nama kelas jika tersedia
    target_idx = None
    names = yolo_result.names if hasattr(yolo_result, "names") else None
    if target_class_name and names:
        for k, v in names.items():
            if str(v).lower() == str(target_class_name).lower():
                target_idx = int(k)
                break

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else None

    selected = None
    if target_idx is not None and cls is not None:
        mask = cls == target_idx
        if mask.any():
            idx = conf[mask].argmax()
            selected = xyxy[mask][idx]

    if selected is None:
        idx = conf.argmax()
        selected = xyxy[idx]

    return tuple(map(float, selected.tolist()))


@st.cache_resource(show_spinner=True)
def load_yolo_detector(config: dict) -> Optional[YoloDetector]:
    yolo_cfg = config.get("yolo", {})
    weights = yolo_cfg.get("weights_path")
    weights_url = yolo_cfg.get("weights_url")

    if not weights:
        return None

    # If a URL is provided and the file doesn't exist locally, download it
    try:
        if (weights_url and isinstance(weights_url, str) and weights_url.startswith(("http://", "https://"))):
            dest = Path(weights)
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                urlretrieve(weights_url, dest.as_posix())
    except Exception:
        # Ignore download errors; Ultralytics can still handle hub names like 'yolov8n.pt'
        pass

    # If a local-looking path is provided but still doesn't exist, skip gracefully
    dest = Path(weights)
    looks_local = ("/" in weights or "\\" in weights or dest.suffix in {".pt", ".pth"})
    if looks_local and not dest.exists() and not (weights.lower().startswith("yolo") and weights.endswith(".pt")):
        st.warning("Bobot YOLO tidak ditemukan. Melewati deteksi dan menggunakan seluruh gambar untuk klasifikasi.")
        return None

    # Let Ultralytics handle hub model names (e.g., 'yolov8n.pt') or valid local paths
    try:
        return YoloDetector(
            weights_path=weights,
            conf_threshold=float(yolo_cfg.get("conf_threshold", 0.25)),
            iou_threshold=float(yolo_cfg.get("iou_threshold", 0.45)),
        )
    except Exception as e:
        st.warning(f"Gagal memuat YOLO: {e}")
        return None


@st.cache_resource(show_spinner=True)
def load_classifier(config: dict, key: str):
    cls_cfg = config.get("classification", {}).get(key, {})
    if not cls_cfg:
        return None
    return build_classifier_from_config(cls_cfg)


def run_pipeline(image: Image.Image, config: dict):
    yolo = load_yolo_detector(config)
    eff = load_classifier(config, "efficientnet")
    mob = load_classifier(config, "mobilenet")

    st.subheader("1) Gambar Masukan")
    st.image(image, caption="Gambar terunggah", use_column_width=True)

    # Deteksi YOLO
    st.subheader("2) Deteksi YOLO")
    crop = None
    if yolo is None:
        st.warning("Bobot YOLO tidak ditemukan. Melewati deteksi dan menggunakan seluruh gambar untuk klasifikasi.")
        crop = image
    else:
        results = yolo.detect(image)
        if not results:
            st.warning("Tidak ada hasil deteksi. Menggunakan seluruh gambar.")
            crop = image
        else:
            res = results[0]
            if res.boxes is None or len(res.boxes) == 0:
                st.warning("Tidak ada kotak terdeteksi. Menggunakan seluruh gambar.")
                crop = image
            else:
                target_name = config.get("yolo", {}).get("target_class_name")
                box = get_best_id_box(res, target_name)
                if box is None:
                    st.warning("Tidak ada kotak yang sesuai. Menggunakan seluruh gambar.")
                    crop = image
                else:
                    x1, y1, x2, y2 = box
                    st.write(f"Kotak terdeteksi: x1={int(x1)}, y1={int(y1)}, x2={int(x2)}, y2={int(y2)}")
                    crop = crop_xyxy(image, box)
                    st.image(crop, caption="Bagian ID yang dipotong", use_column_width=True)

    # Klasifikasi
    st.subheader("3) Klasifikasi")
    if eff is None and mob is None:
        st.info("Model klasifikasi tidak tersedia. Silakan atur bobot pada config.yaml.")
        return

    topk = int(config.get("classification", {}).get("topk", 3))

    cols = st.columns(2)
    if eff is not None:
        with cols[0]:
            labels, probs = eff.predict(crop, topk=topk)
            st.write("Prediksi EfficientNet:")
            for l, p in zip(labels, probs):
                st.write(f"- {l}: {p:.3f}")
    if mob is not None:
        with cols[1]:
            labels, probs = mob.predict(crop, topk=topk)
            st.write("Prediksi MobileNet:")
            for l, p in zip(labels, probs):
                st.write(f"- {l}: {p:.3f}")


def main():
    st.title("Deteksi ID (YOLO) + Klasifikasi (EfficientNet & MobileNet)")
    st.caption("Dibuat oleh: Nurdiansyah Krisna Putra (22.11.4972)")

    config = load_config()

    with st.sidebar:
        st.header("Pengaturan")
        st.write("Ubah config.yaml untuk mengatur bobot model dan opsi.")
        if st.button("Muat ulang pengaturan"):
            load_config.clear()
            st.experimental_rerun()

    uploaded = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        run_pipeline(image, config)
    else:
        st.info("Silakan unggah gambar untuk mulai.")


if __name__ == "__main__":
    main()
