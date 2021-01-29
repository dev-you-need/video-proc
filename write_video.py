import cv2
import deepface

from tqdm import tqdm

import numpy as np
from deepface.extendedmodels import Race
from deepface.extendedmodels import Gender
from deepface.extendedmodels import Age

from tensorflow.keras.preprocessing import image

import mtcnn

from tools import *

input_video = 'video1.mp4'
output_video = 'video1_out3.mp4'
face_min_height_scale = 8

race_model = Race.loadModel()
gender_model = Gender.loadModel()
age_model = Age.loadModel()

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
fps = round(fps)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

face_min_height = int(height//face_min_height_scale)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

face_detector = mtcnn.MTCNN()

def show_faces(img, faces):
    for (x, y, w, h) in faces:

        if min((x, y, w, h)) < 0:
            continue

        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = img[y:y + h, x:x + w]

        race_label = ethnicity_detect(face)
        img = cv2.putText(img, race_label, (int(x), int(y + h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        gender_label = gender_detect(face)
        img = cv2.putText(img, gender_label, (int(x), int(y + h+22)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        age_label = age_detect(face)
        img = cv2.putText(img, str(age_label), (int(x), int(y + h + 44)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img

def extend_bbox(bbox, padding=10):
    (x, y, w, h) = bbox

    if x-padding>0:
        x=x-padding
    else:
        x=0

    if y-padding>0:
        y=y-padding
    else:
        y=0

    w = w + padding * 2

    h = h + padding * 2

    return x, y, w, h

def cut_faces(img, faces):
    imgs = []
    for (x, y, w, h) in faces:
        x, y, w, h = extend_bbox((x, y, w, h))

        imgs.append(img[y:y + h, x:x + w])
    return imgs

def img_preprocess(img, target_size=(224, 224)):
    img = cv2.resize(img, target_size)
    img_pixels = image.img_to_array(img)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    return img_pixels

def ethnicity_detect(img):
    result = 'Unknown'
    race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']

    try:
        img_pixels = img_preprocess(img)

        preds = race_model.predict(img_pixels)
        result = "{}: {:02.0f}%".format(race_labels[np.argmax(preds)], np.max(preds) * 100)
    except Exception as e:
        print(e)

    return result


def age_detect(img):
    apparent_age = 'Unknown'

    try:
        img_pixels = img_preprocess(img)
        preds = age_model.predict(img_pixels)[0,:]
        apparent_age = int(Age.findApparentAge(preds))

    except Exception as e:
        print(e)

    return apparent_age

def gender_detect(img):
    gender = 'Unknown'

    try:
        img_pixels = img_preprocess(img)
        gender_prediction = gender_model.predict(img_pixels)[0,:]

        if np.argmax(gender_prediction) == 0:
            gender = "Woman"
        elif np.argmax(gender_prediction) == 1:
            gender = "Man"

    except Exception as e:
        print(e)

    return gender


def face_detect_mtcnn(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = face_detector.detect_faces(img_rgb)

    faces = []
    if len(detections) > 0:
        for d in detections:
            x, y, w, h = d["box"]

            if (h > height / face_min_height_scale) and d['confidence']>0.9:
                faces.append((x, y, w, h))


    return faces

count = 0
frame_face_count = 0

pbar = tqdm(total=length)

while (cap.isOpened()):
    cap.set(1, count)
    ret, frame = cap.read()

    if ret and (count < 200):
        faces = face_detect_mtcnn(frame)
        frame = show_faces(frame, faces)
        out.write(frame)

        count += fps * 10
        pbar.update()

    else:
        break

cap.release()
out.release()
pbar.close()