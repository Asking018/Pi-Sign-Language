import os
import requests
import zipfile
import shutil

def download_dataset():
    # Project root: parent of model_training
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(_script_dir)
    target_dir = os.path.join(project_root, 'asl_alphabet_test')
    zip_path = os.path.join(project_root, 'asl_dataset.zip')

    # Download URL for the ASL Alphabet Dataset (TensorFlow)
    url = "https://storage.googleapis.com/download.tensorflow.org/data/asl_alphabet_train.zip"

    print("Downloading ASL Alphabet Dataset...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(zip_path, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            for data in response.iter_content(chunk_size=8192):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total_size)
                print(f"\rDownloading: [{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes", end='')

    print("\nExtracting dataset...")
    extract_temp = os.path.join(project_root, '_asl_extract')
    os.makedirs(extract_temp, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_temp)
        # Zip usually contains asl_alphabet_train/ with A, B, C... subdirs
        inner = os.path.join(extract_temp, 'asl_alphabet_train')
        if os.path.isdir(inner):
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
            shutil.move(inner, target_dir)
        else:
            # Zip content is directly A, B, C...
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
            os.makedirs(target_dir, exist_ok=True)
            for name in os.listdir(extract_temp):
                src = os.path.join(extract_temp, name)
                dst = os.path.join(target_dir, name)
                shutil.move(src, dst)
    finally:
        shutil.rmtree(extract_temp, ignore_errors=True)
    os.remove(zip_path)
    print(f"Dataset ready at: {target_dir}")

if __name__ == "__main__":
    download_dataset()
