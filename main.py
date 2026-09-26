import cv2
import time

from camera_stream import CameraStream
from face_engine import FaceEngine
from face_store import FaceStore
from liveness import BlinkLivenessDetector


PROCESS_EVERY_N_FRAME = 2
MATCH_THRESHOLD = 0.5
ENABLE_LIVENESS = True


def draw_results(frame, faces, liveness_ok):
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        name = face["name"] or "Unknown"
        score = face["score"]

        if name != "Unknown" and (not ENABLE_LIVENESS or liveness_ok):
            color = (0, 255, 0)
            label = f"{name} ({score:.2f})"
        elif name != "Unknown" and ENABLE_LIVENESS and not liveness_ok:
            color = (0, 165, 255)
            label = f"{name} - verifikasi liveness..."
        else:
            color = (0, 0, 255)
            label = f"Unknown ({score:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    engine = FaceEngine()
    store = FaceStore()
    liveness = BlinkLivenessDetector() if ENABLE_LIVENESS else None
    stream = CameraStream(source=0).start()

    frame_count = 0
    cached_faces = []
    prev_time = time.time()

    try:
        while True:
            frame = stream.read()
            frame_count += 1

            if frame_count % PROCESS_EVERY_N_FRAME == 0:
                detections = engine.detect_and_embed(frame)
                cached_faces = []
                for det in detections:
                    name, score = store.search(det["embedding"], threshold=MATCH_THRESHOLD)
                    cached_faces.append({
                        "bbox": det["bbox"],
                        "name": name,
                        "score": score if score is not None else 0.0,
                    })

            liveness_ok = True
            if ENABLE_LIVENESS and liveness is not None:
                liveness_ok = liveness.update(frame)

            draw_results(frame, cached_faces, liveness_ok)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Face Recognition v2 - ArcFace + FAISS", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("e"):
                detections = engine.detect_and_embed(frame)
                if detections:
                    name = input("Nama pelanggan/karyawan: ")
                    store.add(name, detections[0]["embedding"])
                    print(f"Terdaftar: {name}")
                else:
                    print("Tidak ada wajah terdeteksi untuk enroll")
                if liveness is not None:
                    liveness.reset()
    finally:
        stream.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
