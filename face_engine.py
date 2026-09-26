import numpy as np
from insightface.app import FaceAnalysis


class FaceEngine:
    def __init__(self, providers=("CPUExecutionProvider",), det_size=(320, 320)):
        self.app = FaceAnalysis(name="buffalo_l", providers=list(providers))
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect_and_embed(self, frame_bgr: np.ndarray):
        faces = self.app.get(frame_bgr)
        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            embedding = face.normed_embedding
            results.append({
                "bbox": (bbox[0], bbox[1], bbox[2], bbox[3]),
                "embedding": embedding,
                "det_score": float(face.det_score),
            })
        return results
