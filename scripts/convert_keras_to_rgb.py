import argparse
import json
import os
import tempfile
from zipfile import ZipFile

import tensorflow as tf


def infer_input_and_classes(config: dict):
    height = width = 224
    channels = 3
    num_classes = None

    # Try to find input shape
    def search_input(conf):
        nonlocal height, width, channels
        if isinstance(conf, dict):
            if conf.get("class_name") in ("InputLayer",):
                cfg = conf.get("config", {})
                shape = cfg.get("batch_input_shape") or cfg.get("batch_shape")
                if isinstance(shape, list) and len(shape) == 4:
                    _, h, w, c = shape
                    if isinstance(h, int):
                        height = h
                    if isinstance(w, int):
                        width = w
                    if isinstance(c, int):
                        channels = c
            for v in conf.values():
                search_input(v)
        elif isinstance(conf, list):
            for v in conf:
                search_input(v)

    # Try to find output units (num classes)
    def search_classes(conf):
        nonlocal num_classes
        if isinstance(conf, dict):
            if conf.get("class_name") in ("Dense",):
                cfg = conf.get("config", {})
                units = cfg.get("units")
                if isinstance(units, int):
                    num_classes = units
            for v in conf.values():
                search_classes(v)
        elif isinstance(conf, list):
            for v in conf:
                search_classes(v)

    search_input(config)
    search_classes(config)

    if num_classes is None:
        num_classes = 2

    return int(height), int(width), int(channels), int(num_classes)


def extract_weights(keras_path: str) -> str:
    with ZipFile(keras_path, 'r') as z:
        weights_member = None
        for name in z.namelist():
            if name.endswith('weights.h5'):
                weights_member = name
                break
        if not weights_member:
            raise RuntimeError('weights.h5 not found inside keras archive')
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.h5')
        os.close(tmp_fd)
        with open(tmp_path, 'wb') as f:
            f.write(z.read(weights_member))
        return tmp_path


def read_config(keras_path: str) -> dict:
    with ZipFile(keras_path, 'r') as z:
        cfg_member = None
        for name in z.namelist():
            base = name.split('/')[-1].lower()
            if base in ('config.json', 'model.json'):
                cfg_member = name
                break
        if not cfg_member:
            raise RuntimeError('config.json/model.json not found in keras archive')
        return json.loads(z.read(cfg_member).decode('utf-8'))


def build_base(model_name: str, num_classes: int) -> tf.keras.Model:
    lname = model_name.lower()
    if 'efficientnet' in lname:
        return tf.keras.applications.efficientnet.EfficientNetB0(include_top=True, weights=None, classes=num_classes)
    if 'mobilenet' in lname and 'v2' in lname:
        return tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=True, weights=None, classes=num_classes)
    if 'mobilenet' in lname:
        return tf.keras.applications.mobilenet.MobileNet(include_top=True, weights=None, classes=num_classes)
    # default
    return tf.keras.applications.efficientnet.EfficientNetB0(include_top=True, weights=None, classes=num_classes)


def convert(keras_in: str, keras_out: str, model_name: str):
    cfg = read_config(keras_in)
    h, w, c, num_classes = infer_input_and_classes(cfg)
    weights_path = extract_weights(keras_in)
    try:
        # Build corrected RGB model
        inp = tf.keras.Input(shape=(h, w, 3))
        base = build_base(model_name, num_classes)
        out = base(inp)
        model = tf.keras.Model(inp, out)
        # Load weights by name, skipping mismatches
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        # Save corrected model
        model.save(keras_out)
        print(f'Saved corrected RGB model to {keras_out} (input: {(h,w,3)}, classes: {num_classes})')
    finally:
        try:
            os.remove(weights_path)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True, help='Input .keras path')
    ap.add_argument('--out', dest='out', required=True, help='Output .keras path')
    ap.add_argument('--arch', dest='arch', required=True, help='Architecture name (efficientnet_b0, mobilenet_v2, etc.)')
    args = ap.parse_args()
    convert(args.inp, args.out, args.arch)


if __name__ == '__main__':
    main()
