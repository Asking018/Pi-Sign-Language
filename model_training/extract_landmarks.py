"""
Extract hand landmarks from ASL training images using MediaPipe.
Output: landmarks.csv for training a classifier (more reliable than image-based CNN).
"""
import os
import csv
import numpy as np
from pathlib import Path

# Use mediapipe for hand detection
try:
    import mediapipe as mp
except ImportError:
    print("Install mediapipe: pip install mediapipe")
    exit(1)

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
DATASET_PATH = os.environ.get('ASL_DATASET_PATH', os.path.join(_project_root, 'asl_alphabet_test'))
OUTPUT_CSV = os.path.join(_project_root, 'models', 'landmarks_data.csv')
SAMPLES_PER_CLASS = int(os.environ.get('SAMPLES_PER_CLASS', '300'))  # Use 300 per letter for speed

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    classes = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        # Store x,y only (42 values) to match frontend MediaPipe output
        header = ['class'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)]
        writer.writerow(header)

        try:
            import cv2
        except ImportError:
            print("Install opencv: pip install opencv-python")
            return

        for cls in classes:
            cls_path = os.path.join(DATASET_PATH, cls)
            images = [p for p in Path(cls_path).rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
            np.random.shuffle(images)
            count = 0
            for img_path in images:
                if count >= SAMPLES_PER_CLASS:
                    break
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0]
                    xs = [lm.landmark[i].x for i in range(21)]
                    ys = [lm.landmark[i].y for i in range(21)]
                    # Relative to wrist (index 0) so shape matches frontend
                    base_x, base_y = xs[0], ys[0]
                    xs_rel = [x - base_x for x in xs]
                    ys_rel = [y - base_y for y in ys]
                    row = [cls] + xs_rel + ys_rel  # 42 values, same as frontend
                    writer.writerow(row)
                    count += 1
            print(f"{cls}: {count} samples")
    
    hands.close()
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
