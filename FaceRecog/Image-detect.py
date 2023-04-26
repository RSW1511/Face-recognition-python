import face_recognition as fr
import cv2 as cv
from tkinter import Tk
from tkinter.filedialog import askopenfilename # open filechooser
import os

# choose photo
Tk().withdraw()
load_image = askopenfilename()
target_image = fr.load_image_file(load_image)
target_encoding = fr.face_encodings(target_image)


# Searches for occurrence of this face
def encode_faces(folder):
    list_Faces_encoding = []

    for filename in os.listdir(folder):
        known_image = fr.load_image_file(f'{folder}{filename}')
        know_encoding = fr.face_encodings(known_image)[0]

        list_Faces_encoding.append((know_encoding, filename))
    return list_Faces_encoding


def find_target_face():
    face_location = fr.face_locations(target_image)

    for person in encode_faces('Faces/'):
        encoded_face = person[0]
        filename = person[1]

        is_target_face = fr.compare_faces(encoded_face, target_encoding, tolerance=0.55)
        print(f'{is_target_face} {filename}')

        if face_location:
            face_number = 0
            for location in face_location:
                if is_target_face[face_number]:
                    label = os.path.splitext(filename)[0]  # remove the file extension
                    create_frame(location, label)
                face_number += 1


def create_frame(location, label):
    top, right, bottom, left = location
    cv.rectangle(target_image, (left, top), (right, bottom), (255, 0, 0), 2)
    cv.rectangle(target_image, (left, bottom + 20), (right, bottom), (255, 0, 0), cv.FILLED)
    cv.putText(target_image, label, (left + 3, bottom + 14), cv.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)


def render_image():
    rgb_img = cv.cvtColor(target_image, cv.COLOR_BGR2RGB)
    cv.imshow('Image Face Recognition', rgb_img)
    cv.waitKey(0)


find_target_face()
render_image()
