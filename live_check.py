import cv2
import mediapipe as mp
import numpy as np
from keras_facenet import FaceNet

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
embedder = FaceNet()

# Load reference embedding
ref_embedding = np.load("employee_embedding.npy")

# EAR blink detection
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(lm, eye_points, w, h):
    p = lambda i: np.array([lm[eye_points[i]].x * w, lm[eye_points[i]].y * h])
    vertical1 = np.linalg.norm(p(1) - p(5))
    vertical2 = np.linalg.norm(p(2) - p(4))
    horizontal = np.linalg.norm(p(0) - p(3))
    return (vertical1 + vertical2) / (2.0 * horizontal)

def get_face_embedding(frame):
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as detector:
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None, None
        h, w, _ = frame.shape
        bbox = results.detections[0].location_data.relative_bounding_box
        x1, y1 = int(bbox.xmin*w), int(bbox.ymin*h)
        x2, y2 = int((bbox.xmin+bbox.width)*w), int((bbox.ymin+bbox.height)*h)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None, None
        emb = embedder.embeddings([face])[0]
        return emb, (x1, y1, x2, y2)

cap = cv2.VideoCapture(0)
blink_count = 0

with mp_face_mesh.FaceMesh(max_num_faces=1) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Blink detection
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            leftEAR = eye_aspect_ratio(lm, LEFT_EYE, w, h)
            rightEAR = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
            EAR = (leftEAR + rightEAR) / 2.0
            if EAR < 0.2:  # Blink threshold
                blink_count += 1
                cv2.putText(frame, "Blink Detected", (50,50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Face recognition
        emb, bbox = get_face_embedding(frame)
        if emb is not None:
            sim = np.dot(ref_embedding, emb) / (np.linalg.norm(ref_embedding)*np.linalg.norm(emb))
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"Sim: {sim:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            if sim > 0.75 and blink_count > 0:
                cv2.putText(frame, " LIVE FACE VERIFIED", (50,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)

        cv2.imshow("Live Face Check", frame)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
