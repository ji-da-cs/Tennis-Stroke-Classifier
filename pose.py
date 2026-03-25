import cv2
import numpy as np

BODY_PARTS = {
    0: "Head", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist",
    5: "LShoulder", 6: "LElbow", 7: "LWrist", 8: "RHip", 9: "RKnee",
    10: "RAnkle", 11: "LHip", 12: "LKnee", 13: "LAnkle", 14: "Chest"
}


def angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos = np.clip(cos, -1, 1)
    return np.degrees(np.arccos(cos))


def get_keypoints(net, frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()
    H, W = output.shape[2], output.shape[3]
    points = []
    for i in range(len(BODY_PARTS)):
        probMap = output[0, i, :, :]
        _, _, _, point = cv2.minMaxLoc(probMap)
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        points.append((int(x), int(y)))
    return points
