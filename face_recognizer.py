import face_recognition
import cv2
import os
import glob
import numpy as np


class FaceRecognizer:
    def __init__(self):
        self.known_faces_encodings = []
        self.known_faces_names = []
        self.resize_factor = 0.25

    def load_images_encoding(self, path):
        people_names = os.listdir(path)
        people_names = list(filter(lambda name: os.path.isdir(path + name), people_names))
        print(f"{len(people_names)} people folders found.")

        # Load images for every person
        for person_name in people_names:
            print(f"Loading images for {person_name}")

            person_images_path = glob.glob(os.path.join(path + person_name, "*.*"))
            # Store image encoding and names
            for image_path in person_images_path:
                print(f"    Loading {image_path}")
                img = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Get encoding
                img_encoding = face_recognition.face_encodings(img_rgb)
                img_encoding = img_encoding[0]

                self.known_faces_encodings.append(img_encoding)
                self.known_faces_names.append(person_name)

            print(f"  ✓ Loading images for {person_name} succeed")
        print("Images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        # Find all the faces and face encodings in the current frame of video
        # Convert image colors from BGR (OpenCV) to RGB (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_faces_encodings, face_encoding)
            # See if the face is a match for the known face(s)
            name = "Unknown"
            if len(matches) > 0:
                # Select known face with the smallest distance to the detected one
                face_distances = face_recognition.face_distance(self.known_faces_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_faces_names[best_match_index]
            face_names.append(name)

        # Adjust coordinates with resize factor
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.resize_factor

        return face_locations.astype(int), face_names
