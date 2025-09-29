import os
import cv2
import numpy as np
import json
from ultralytics import YOLO

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
SNAKE_DATA_FILE = "snake_data.json"

# Load YOLO models
det_model = YOLO('models/snake_detection.pt')
cls_model = YOLO('models/snake_classification.pt')

def load_snake_data():
    with open(SNAKE_DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def run_pipeline(img_path, filename):
    img = cv2.imread(img_path)
    results = det_model.predict(img_path)[0]

    pretty_name = "Unknown"
    
    if results.obb is not None and len(results.obb.xyxyxyxy) > 0:
        for i, obb_pts in enumerate(results.obb.xyxyxyxy.cpu().numpy()):
            points = np.array(obb_pts, dtype=np.float32).reshape((4, 2))
            width = int(np.linalg.norm(points[0] - points[1]))
            height = int(np.linalg.norm(points[1] - points[2]))
            dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(points, dst_pts)
            warped = cv2.warpPerspective(img, M, (width, height))

            crop_path = os.path.join(UPLOAD_FOLDER, f"crop_{i}.jpg")
            cv2.imwrite(crop_path, warped)
            cls_results = cls_model.predict(crop_path)[0]

            if cls_results.probs is not None:
                class_id = int(cls_results.probs.top1)
                class_name = cls_model.names[class_id]
                confidence = float(cls_results.probs.top1conf)

                # Draw box
                pretty_name = class_name.replace("-", " ").title()
                label = f"{pretty_name} ({confidence:.2f})"
                int_points = points.astype(int)

                for j in range(4):
                    pt1 = tuple(int_points[j])
                    pt2 = tuple(int_points[(j + 1) % 4])
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)

                # # Label text
                # top_mid_x = int((int_points[0][0] + int_points[1][0]) / 2)
                # top_mid_y = int((int_points[0][1] + int_points[1][1]) / 2)
                # label_x = top_mid_x - int(cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] / 2)
                # label_y = top_mid_y - 10
                # cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    result_path = os.path.join(RESULT_FOLDER, f"result_{filename}")
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    cv2.imwrite(result_path, img)

    rel_result_path = os.path.relpath(result_path, "static").replace("\\", "/")
    return rel_result_path, pretty_name
