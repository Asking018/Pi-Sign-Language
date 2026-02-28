# Pi Sign Language Translator – Copilot Instructions

This repository is a single‑page web app backed by a small Flask server and a couple of
machine‑learning models.  The goal is to translate American Sign Language letters into
text in real time.  Below are the key pieces of information that a code‑writing AI
needs to be productive.

---

## Big‑picture architecture

* **Backend**: `app.py` is a self‑contained Flask application.  It serves
  static assets, exposes `/predict` for inference and a few simple document pages
  (`/`, `/learn`, `/premium`).  CORS is open so the frontend can run on localhost or
a remote dev server.
* **Models** live under `models/`.  Two types are used:
  * `best_model.h5` – image CNN trained on the ASL alphabet (see
    `model_training/train_model.py`).  Loaded with `tf.keras.models.load_model`
  * `landmark_classifier.joblib` – RandomForest trained on 21‑point hand landmarks
    (see `model_training/train_landmark_model.py` and `extract_landmarks.py`).
* **Frontend**: plain HTML/JS in `static/` and `templates/`.  `static/app.js` drives
  camera access, MediaPipe hand detection, smoothing, and sends either a base64
  crop or landmark array to the server.  Prediction smoothing and the word‑creator
  logic live in `wordLogic.js`.
* **Pi Network**: the app uses the Pi SDK for authentication/payments.  Most pages
  include a stub/mock in `templates/index.html` so the UI works without the
  real SDK during local development.  Payments to unlock the "Word Creator" are
  handled by frontend JS (`auth.js`, `app.js`) calling the Pi SDK; the backend
  only needs `PI_SERVER_API_KEY` (env) to approve/complete transactions according
  to the documentation in `docs/PI_PLATFORM_API.md`.

Data flow:

```
camera -> tfjs/MediaPipe detector -> landmarks/base64 -> POST /predict
    -> (CNN or landmark model -> letter,confidence) -> JSON response -> client
    -> wordLogic buffers & locks in letters -> UI
```

Geometric heuristics in `app.py` override some confusable letters (A, H, M, C).
When a prediction is made the server will also write a debug png to
`static/hand_debug.png` so you can inspect what it saw.

---

## Developer workflows

1. **Run the server**
   ```bash
   pip install -r requirements.txt
   export PI_SERVER_API_KEY=...     # optional if using payments
   python app.py                   # listens on 127.0.0.1:5000
   ```
   Browse to `http://localhost:5000`.

2. **Train or update the CNN**
   * Put the ASL alphabet dataset under `asl_alphabet_test/` or set
     `ASL_DATASET_PATH` env var.
   * `python model_training/train_model.py` (tweak `ASL_EPOCHS` environment
     variable for quick runs).  The best model is saved to `models/best_model.h5`.
   * After training run `python convert_model.py` to generate a TensorFlow SavedModel
     under `static/model` (used by the frontend if you ever migrate inference
     client‑side).  `convert_to_onnx.py` is handy if you want an ONNX export.

3. **Train or update landmark classifier**
   * `python model_training/extract_landmarks.py` will dump normalized landmarks to
     `models/landmarks_data.csv` (adjust `SAMPLES_PER_CLASS` through env var).
   * Run `python model_training/train_landmark_model.py`; produces
     `models/landmark_classifier.joblib` which `app.py` will load at startup.

4. **Testing predictions**
   * Use `model_training/test_model.py` with images in `test_images/`.
   * The UI has a "Test with 'A' Sign" button that injects a hard‑coded landmark
     array.
   * `check_signs.py` displays sample dataset images and can generate an
     "improved" hand graphic for debugging mis‑classifications.

> **Note:** there are no automated unit tests in this repo; most validation
> happens by running the above scripts or exercising the web UI.

---

## Project‑specific conventions

* Classes are always the 26 letters plus a space.  The mapping file
  `models/index_to_class.json` is fetched by the frontend; do not change the order
  without updating any hard‑coded arrays in `app.js`/`wordLogic.js`.
* The `/predict` API accepts one of two payload shapes:

```json
{ "image_base64": "data:image/png;base64,..." }
```

or

```json
{ "landmarks": [x0,y0, x1,y1, ..., x20,y20] }
```

Response example:

```json
{
  "letter": "A",
  "confidence": 0.87,
  "method": "landmark",        // or "cnn"
  "debug_image": "/static/hand_debug.png",
  "message": "Low confidence" // optional
}
```

  Missing or invalid data triggers HTTP 400/500 with an `error` field.

* Prediction smoothing is done entirely client‑side (see `PREDICTION_HISTORY_SIZE`
  and `getDominantLetter` in `wordLogic.js`).  The word‑creator uses a 3‑second
  lock‑in timer and is gated by `isPremium`/`localStorage.pi_premium`.
* Environment variables drive configuration:
  * `PI_SERVER_API_KEY` – unlocks server payment endpoints documented in `docs/`
  * `ASL_DATASET_PATH`, `ASL_EPOCHS`, `SAMPLES_PER_CLASS` – training helpers
  * Any other variables are ignored by the Python code and can be safely added
    for CI or deployment.

* Static assets are served via `send_from_directory`.  New JS/CSS files should
  live under `static/` and be referenced from `templates/index.html` (or other
  HTML files in `templates/`).

* Logging uses Python's `logging` module; start the server with `DEBUG` enabled to
  see the heuristics in action.

---

## Integration & external dependencies

* **Machine Learning:** TensorFlow 2.x (CPU‑only) on the backend.  Frontend uses
  TensorFlow.js and MediaPipe hands (see CDN links in `templates/index.html`).
* **Pi Network SDK:** frontend script loaded from `sdk.minepi.com`; backend only
  makes simple HTTPS requests to `https://api.minepi.com` if `PI_SERVER_API_KEY`
  is set.  Consult `docs/PI_PLATFORM_API.md` for the exact endpoints and request
  shapes.
* **CORS** is already configured; no special reverse proxy rules are required
  during development.

---

If anything above seems unclear or if your change requires diving into a new
area of the code, ask for clarification – we can iterate on these instructions.
