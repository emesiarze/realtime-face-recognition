import cv2
from face_recognizer import FaceRecognizer

frame_width = 3
text_height = 36
text_color = (0, 0, 0)
frame_color = (0, 200, 0)
label_font = cv2.FONT_HERSHEY_DUPLEX
images_directory = "images"

# Load known faces
sfr = FaceRecognizer()
sfr.load_images_encoding(f"{images_directory}\\")

# Access video source
cap = cv2.VideoCapture(0)

while True:
    if not cap.isOpened():
        print("Could not access video source.")
        exit(1)

    # Start reading from video source
    ret, frame = cap.read()

    # Detect faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        # Put the frame around face
        cv2.rectangle(frame, (x1, y1), (x2, y2), frame_color, frame_width)
        # Put the label with background
        cv2.rectangle(frame, (x1 - frame_width, y1 - text_height), (x2 + frame_width, y1), frame_color, -1)
        cv2.putText(frame, name, (x1, y1 - 10), label_font, .7, text_color, 2)

    cv2.imshow("Face recognition", frame)

    # Press any key to exit
    key = cv2.waitKey(100)
    if key != -1:
        break

cap.release()
cv2.destroyAllWindows()
