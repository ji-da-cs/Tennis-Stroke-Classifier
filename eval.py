import sys
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from model import DinoV3Linear
from pose import BODY_PARTS, angle, get_keypoints
from reference import compute_reference_angles

STROKES = {
    0: "backhand_back",
    1: "backhand_front",
    2: "forehand_back",
    3: "forehand_front",
    4: "serve_front",
    5: "serve_back",
}

MODEL_NAME = "./dinov3-vitb16-pretrain-lvd1689m"
POSE_PROTO = "models/pose_deploy_linevec_faster_4_stages.prototxt"
POSE_WEIGHTS = "models/pose_iter_160000.caffemodel"
CLASSIFIER_WEIGHTS = "weights/model_best.pt"


def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    model = DinoV3Linear(backbone, num_classes=6, freeze_backbone=True).to(device)
    checkpoint = torch.load(CLASSIFIER_WEIGHTS, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    net = cv2.dnn.readNetFromCaffe(POSE_PROTO, POSE_WEIGHTS)
    return model, image_processor, net, device


def evaluate_stroke(filepath, model, image_processor, net, refs, device):
    img = Image.open(filepath).convert("RGB")
    inputs = image_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(inputs["pixel_values"])
    pred = torch.softmax(logits, dim=-1).argmax(dim=-1).item()
    stroke = STROKES[pred]
    print(f"Stroke Prediction: {stroke}")

    frame = cv2.imread(filepath)
    points = get_keypoints(net, frame)
    ref = refs[stroke]
    score = 0.0

    if stroke in ("backhand_back", "backhand_front"):
        a_left = angle(points[5], points[6], points[7])
        a_right = angle(points[2], points[3], points[4])
        print(f"Left arm angle: {a_left:.1f}, suggested: {ref['avg_left']:.1f}")
        print(f"Right arm angle: {a_right:.1f}, suggested: {ref['avg_right']:.1f}")
        score = ((360 - abs(a_left - ref["avg_left"])) / 360) / 2 + ((360 - abs(a_right - ref["avg_right"])) / 360) / 2
    elif stroke in ("forehand_back", "forehand_front"):
        a_right = angle(points[2], points[3], points[4])
        print(f"Right arm angle: {a_right:.1f}, suggested: {ref['avg_right']:.1f}")
        score = (360 - abs(a_right - ref["avg_right"])) / 360
    elif stroke in ("serve_back", "serve_front"):
        a_right = angle(points[2], points[3], points[4])
        a_body = angle(points[2], points[8], points[10])
        print(f"Right arm angle: {a_right:.1f}, suggested: {ref['avg_right']:.1f}")
        print(f"Body angle: {a_body:.1f}, suggested: {ref['avg_body']:.1f}")
        score = ((360 - abs(a_right - ref["avg_right"])) / 360) / 2 + ((360 - abs(a_body - ref["avg_body"])) / 360) / 2

    print(f"Score: {score:.4f}")
    return score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval.py <image_path>")
        sys.exit(1)
    model, image_processor, net, device = load_models()
    refs = compute_reference_angles(net)
    evaluate_stroke(sys.argv[1], model, image_processor, net, refs, device)
