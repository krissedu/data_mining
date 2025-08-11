# ID Detection + Classification (Streamlit)

This app lets you upload an image, detects an ID document using YOLO, crops the detected box, then classifies the crop using EfficientNet and MobileNet.

## Features
- YOLO object detection for ID document bounding box
- Automatic crop of the detected region
- Dual classification heads (EfficientNet and MobileNet) with top-k predictions

## Setup
1. Create a virtual environment and install requirements:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Place your trained weight files:
   - YOLO ID detection weights at `weights/yolo_id.pt`
   - EfficientNet fine-tuned weights at `weights/efficientnet_b0_idcls.pth`
   - MobileNet fine-tuned weights at `weights/mobilenetv3_large_idcls.pth`

3. Edit `config.yaml` with correct paths, class names, and thresholds.

## Run
```bash
streamlit run app.py
```

Open the URL shown in your terminal (usually `http://localhost:8501`).

## Notes
- If any model weights are missing, the app will gracefully skip that step and notify you.
- If your YOLO class name for the ID document is different from `id`, change `yolo.target_class_name` in `config.yaml`.
- If your fine-tuned classifiers use a different number of classes or order, update `num_classes` and `class_names` accordingly.
