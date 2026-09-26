import cv2
from pathlib import Path

from face_engine import FaceEngine
from face_store import FaceStore


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def enroll_from_directory(dataset_dir, engine, store, images_per_person_limit=None):
    dataset_path = Path(dataset_dir)

    for person_dir in sorted(dataset_path.iterdir()):
        if not person_dir.is_dir():
            continue

        name = person_dir.name
        image_files = sorted(
            p for p in person_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images_per_person_limit:
            image_files = image_files[:images_per_person_limit]

        enrolled_count = 0
        for img_path in image_files:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  Gagal dibaca: {img_path.name}")
                continue

            detections = engine.detect_and_embed(frame)
            if not detections:
                print(f"  Tidak ada wajah terdeteksi: {img_path.name}")
                continue

            best = max(detections, key=lambda d: d["det_score"])
            store.add(name, best["embedding"], persist=False)
            enrolled_count += 1

        print(f"{name}: {enrolled_count}/{len(image_files)} foto berhasil di-enroll")

    store.save()
    

if __name__ == "__main__":
    engine = FaceEngine()
    store = FaceStore()
    enroll_from_directory("dataset", engine, store)
