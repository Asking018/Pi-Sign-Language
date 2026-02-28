"""
Train a landmark-based ASL classifier (Random Forest).
Uses landmarks extracted by extract_landmarks.py - more robust to lighting/background.
"""
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
CSV_PATH = os.path.join(_project_root, 'models', 'landmarks_data.csv')
MODEL_PATH = os.path.join(_project_root, 'models', 'landmark_classifier.joblib')

def normalize_landmarks(X):
    """Center on wrist (x0,y0) and scale by hand size. X shape: (n, 42)."""
    X = np.array(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    wrist_x = X[:, 0:1]
    wrist_y = X[:, 21:22]
    xs = X[:, :21] - wrist_x
    ys = X[:, 21:42] - wrist_y
    scale = np.sqrt(np.mean(xs**2 + ys**2, axis=1, keepdims=True))
    scale = np.maximum(scale, 1e-6)
    xs = xs / scale
    ys = ys / scale
    return np.hstack([xs, ys])

def main():
    if not os.path.exists(CSV_PATH):
        print(f"Run extract_landmarks.py first to create {CSV_PATH}")
        return
    
    import csv
    rows = []
    with open(CSV_PATH) as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if len(row) < 43:
                continue
            cls = row[0]
            vals = [float(x) for x in row[1:43]]  # 42 values: 21*x + 21*y
            if len(vals) != 42:
                continue
            rows.append((cls, vals))
    
    X = np.array([r[1] for r in rows])
    y = np.array([r[0] for r in rows])
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Normalize
    X = normalize_landmarks(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    
    clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"Test accuracy: {acc:.2%}")
    
    joblib.dump({'model': clf, 'classes': le.classes_.tolist()}, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()
