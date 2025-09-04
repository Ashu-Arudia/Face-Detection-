import cv2
import mediapipe as mp
import numpy as np
from keras_facenet import FaceNet

mp_face_detection = mp.solutions.face_detection
embedder = FaceNet()

def get_face_embedding(frame):
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as detector:
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None
        h, w, _ = frame.shape
        bbox = results.detections[0].location_data.relative_bounding_box
        x1, y1 = int(bbox.xmin*w), int(bbox.ymin*h)
        x2, y2 = int((bbox.xmin+bbox.width)*w), int((bbox.ymin+bbox.height)*h)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        emb = embedder.embeddings([face])[0]
        return emb

cap = cv2.VideoCapture(0)
print("📷 Press 's' to capture face, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Enrollment", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        emb = get_face_embedding(frame)
        if emb is not None:
            np.save("employee_embedding.npy", emb)
            print(" Reference face embedding saved!")
        else:
            print(" No face detected, try again")
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
