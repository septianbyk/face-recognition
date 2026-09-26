# Face Recognition System

Real-time face recognition pipeline built for a application, using pre-trained deep learning face embeddings (ArcFace via InsightFace), FAISS for scalable similarity search, and blink-based liveness detection to reduce photo-spoofing risk.

## Features

- Real-time webcam capture with a dedicated capture thread (no frame-drop blocking)
- Face detection + 512-d ArcFace embedding via InsightFace (`buffalo_l`)
- Cosine-similarity matching against a FAISS index for fast lookup even with large enrolled datasets
- Batch enrollment from a labeled photo dataset (`dataset/<name>/*.jpg`)
- Blink-based liveness check (MediaPipe Face Landmarker, Tasks API) to reject static photo spoofing
- Threshold-based accept/reject decision suitable for FAR/FRR evaluation

## Architecture

```
CameraStream (thread) --> main.py loop --> FaceEngine (detect + embed)
                                        --> FaceStore (FAISS cosine match)
                                        --> BlinkLivenessDetector (liveness gate)
                                        --> annotated frame (cv2.imshow)
```

## Project Structure

```
.
├── main.py              # Real-time recognition loop
├── batch_enroll.py       # Bulk enrollment from a labeled dataset folder
├── face_engine.py        # InsightFace detection + embedding wrapper
├── face_store.py          # FAISS-backed embedding storage and search
├── camera_stream.py        # Threaded webcam capture
├── liveness.py              # Blink-based liveness detection
├── requirements.txt
└── dataset/                  # Labeled enrollment photos (see below)
```

## Installation

```
pip install -r requirements.txt
```

The ArcFace model (`buffalo_l`) and the MediaPipe Face Landmarker model are downloaded automatically on first run; an internet connection is required for that first run only.

## Dataset Labeling Convention

One folder per person, folder name = label:

```
dataset/
├── Septian_Bayu/
│   ├── img1.jpg
│   └── img2.jpg
└── Ani_Wijaya/
    └── img1.jpg
```

For accuracy evaluation (FAR/FRR), separate a test set into `genuine/` (enrolled people) and `impostor/` (people not in the system):

```
test_set/
├── genuine/
│   └── Septian_Bayu/
└── impostor/
    └── people_x/
```

## Usage

**1. Enroll a dataset in bulk:**

```
python batch_enroll.py
```

This populates `face_index.faiss` and `face_meta.pkl` in the project directory.

**2. Run real-time recognition:**

```
python main.py
```

- Green box + name = recognized match
- Orange box = matched but awaiting liveness confirmation (blink)
- Red box + "Unknown" = no match found
- Press `e` during the session to enroll a new face on the fly
- Press `q` to quit

## Configuration

Key tunables live at the top of `main.py`:

| Variable | Purpose | Default |
|---|---|---|
| `PROCESS_EVERY_N_FRAME` | Run detection every N frames for FPS optimization | `2` |
| `MATCH_THRESHOLD` | Cosine similarity threshold for accepting a match | `0.5` |
| `ENABLE_LIVENESS` | Toggle the blink-based liveness gate | `True` |

## Known Limitations

- Blink-based liveness is a heuristic and can be bypassed by a replayed video of a genuine blink; it is not a substitute for certified anti-spoofing hardware (e.g. IR/depth cameras) in a production payment system.
- `face_recognition`/dlib is not used in this version; if comparing embedding models for a research writeup, note that ArcFace (this pipeline) generally outperforms dlib's 128-d embedding on pose and low-light robustness.
