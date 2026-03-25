import os
import cv2
import numpy as np
from pose import angle, get_keypoints

STROKE_DIRS = {
    "backhand_back": "backhand_back",
    "backhand_front": "backhand_front",
    "forehand_back": "forehand_back",
    "forehand_front": "forehand_front",
    "serve_back": "serve_back",
    "serve_front": "serve_front",
}


def compute_reference_angles(net):
    refs = {}
    for stroke, directory in STROKE_DIRS.items():
        angles_left, angles_right, angles_body = [], [], []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            frame = cv2.imread(filepath)
            if frame is None:
                continue
            points = get_keypoints(net, frame)
            a_left = angle(points[5], points[6], points[7])
            a_right = angle(points[2], points[3], points[4])
            a_body = angle(points[2], points[8], points[10])
            if not np.isnan(a_left):
                angles_left.append(a_left)
            if not np.isnan(a_right):
                angles_right.append(a_right)
            if not np.isnan(a_body):
                angles_body.append(a_body)
        refs[stroke] = {
            "avg_left": sum(angles_left) / len(angles_left) if angles_left else None,
            "avg_right": sum(angles_right) / len(angles_right) if angles_right else None,
            "avg_body": sum(angles_body) / len(angles_body) if angles_body else None,
        }
    return refs
