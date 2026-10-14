import os
import sys
import json
from pathlib import Path

# Fix Windows console UTF-8 encoding issue for PyTorch prints
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from torchvision import models

def export_to_onnx(
    model_path="models/best_mobilenet_v3.pth",
    output_onnx="models/flood_mobilenetv3.onnx",
    classes_json="models/class_indices.json"
):
    print("=" * 60)
    print("EKSPOR MODEL KE FORMAT ONNX")
    print("=" * 60)

    # 1. Buka mapping kelas
    if Path(classes_json).exists():
        with open(classes_json, "r") as f:
            class_to_idx = json.load(f)
        num_classes = len(class_to_idx)
    else:
        num_classes = 2

    # 2. Re-create model
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Dummy input: batch_size=1, 3 channels, 224x224
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    # 4. Export ONNX (gunakan opset 18 dan fallback legacy jika perlu)
    Path(output_onnx).parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            },
            dynamo=False
        )
    except Exception as e:
        print(f"[RETRY] Ekspor standar gagal: {e}, mencoba dengan dynamo=False...")
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamo=False
        )

    size_mb = os.path.getsize(output_onnx) / (1024 * 1024)
    print(f"[SUKSES] Model berhasil diekspor ke: {output_onnx}")
    print(f"         Ukuran file ONNX: {size_mb:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    export_to_onnx()
