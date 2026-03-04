"""
Create a minimal ASL-alphabet folder structure with placeholder images
so train_model.py can run and produce models/best_model.h5.
Replace with real data (e.g. from Kaggle) and retrain for production.
"""
import os
from PIL import Image, ImageDraw, ImageFont

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
TARGET_DIR = os.path.join(_project_root, 'asl_alphabet_test')
IMG_SIZE = 160
# Minimal samples per class so training runs (use 30+ so validation_split has room)
SAMPLES_PER_CLASS = 40
LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]


def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    for letter in LETTERS:
        class_dir = os.path.join(TARGET_DIR, letter)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(SAMPLES_PER_CLASS):
            # Simple placeholder: colored patch + letter label (varied slightly per sample)
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(30 + i % 50, 60, 90))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 80)
            except OSError:
                font = ImageFont.load_default()
            # Center letter
            bbox = draw.textbbox((0, 0), letter, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (IMG_SIZE - w) // 2
            y = (IMG_SIZE - h) // 2
            draw.text((x, y), letter, fill=(255, 255, 255), font=font)
            # Slight variation: add noise so images aren't identical
            if i % 3 == 0:
                for _ in range(20):
                    draw.point((i * 3 % IMG_SIZE, (i * 5 + _) % IMG_SIZE), fill=(100, 100, 100))
            path = os.path.join(class_dir, f'{letter}_{i:04d}.jpg')
            img.save(path)
        print(f"Created {SAMPLES_PER_CLASS} placeholders for class {letter}")
    print(f"Minimal dataset ready at: {TARGET_DIR}")


if __name__ == "__main__":
    main()
