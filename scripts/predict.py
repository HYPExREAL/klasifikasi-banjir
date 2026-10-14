import sys
import json
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms, models

def predict_single_image(
    image_path,
    model_path="models/best_mobilenet_v3.pth",
    classes_json="models/class_indices.json",
    device_name=None
):
    if not Path(image_path).exists():
        print(f"[ERROR] Gambar tidak ditemukan: {image_path}")
        return None

    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classes
    with open(classes_json, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Model
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(class_to_idx))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    result = {
        "file": str(image_path),
        "predicted_label": idx_to_class[pred_idx],
        "is_flood": idx_to_class[pred_idx] == "banjir",
        "confidence": round(confidence, 4),
        "all_probabilities": {idx_to_class[i]: round(probs[i].item(), 4) for i in range(len(idx_to_class))}
    }

    print("\n" + "=" * 50)
    print("HASIL PREDIKSI GAMBAR:")
    print(f"File             : {image_path}")
    print(f"Prediksi         : {result['predicted_label'].upper()}")
    print(f"Tingkat Keyakinan: {result['confidence'] * 100:.2f}%")
    print(f"Status Banjir    : {'YA (BANJIR)' if result['is_flood'] else 'TIDAK (BUKAN BANJIR)'}")
    print("Probabilitas:")
    for cls_name, prob in result['all_probabilities'].items():
        print(f"  - {cls_name:<15}: {prob*100:.2f}%")
    print("=" * 50)

    return result

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_arg = sys.argv[1]
    else:
        # Default test sample
        sample_img = next(Path("dataset/test/banjir").glob("*.jpg"), None) or next(Path("dataset/test/banjir").glob("*.*"), None)
        img_arg = str(sample_img) if sample_img else "sample.jpg"

    predict_single_image(img_arg)
