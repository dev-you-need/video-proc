import cv2

from tqdm import tqdm

import numpy as np
from deepface.extendedmodels import Race
from deepface.extendedmodels import Gender
from deepface.extendedmodels import Age

from tensorflow.keras.preprocessing import image

import face_recognition
import mtcnn

input_video = 'video4.mp4'
output_video = 'video4_out3.mp4'
face_min_height_scale = 17

race_model = Race.loadModel()
gender_model = Gender.loadModel()
age_model = Age.loadModel()

face_detector = mtcnn.MTCNN()

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
fps = round(fps)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

face_min_height = int(frame_height // face_min_height_scale)

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))


def show_faces(img, faces):
    for (x, y, w, h) in faces:

        if min((x, y, w, h)) <= 0:
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


def face_attrib_recognition(img, faces):
    faces_attr = []

    for (x, y, w, h) in faces:
        if min((x, y, w, h)) < 0:
            continue

        d = {'bbox': (x, y, w, h)}

        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = img[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        race_label = ethnicity_detect(face)
        d['race'] = race_label

        gender_label = gender_detect(face)
        d['gender'] = gender_label

        age_label = age_detect(face)
        d['age'] = age_label

        faces_attr.append(d)
    return faces_attr


def put_text(img, faces_attr):
    for d in faces_attr:
        (x, y, w, h) = d.get('bbox')

        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        race_label = d.get('race')
        img = cv2.putText(img, race_label, (int(x), int(y + h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        gender_label = d.get('gender')
        img = cv2.putText(img, gender_label, (int(x), int(y + h+22)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        age_label = d.get('age')
        img = cv2.putText(img, str(age_label), (int(x), int(y + h + 44)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img


def extend_bbox(bbox, padding=0.6):
    (x, y, w, h) = bbox

    w = max(w, h)
    h = max(w, h)

    x = max(0, x - int(w * padding))
    y = max(0, y - int(h * padding))

    w = min(frame_width - x, w + int(w * padding * 2))
    h = min(frame_height - y, h + int(h * padding * 2))

    return x, y, w, h


def img_preprocess(img, target_size=(224, 224)):
    img = cv2.resize(img, target_size)
    img_pixels = image.img_to_array(img)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255
    return img_pixels


def ethnicity_detect(img):
    result = 'Unknown'
    race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino']

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
        apparent_age = "Age: {}".format(int(Age.findApparentAge(preds)))

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

        gender = '{}: {:02.0f}%'.format(gender, np.max(gender_prediction)*100)

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

            if (h > face_min_height) and d['confidence'] > 0.98:
                x, y, w, h = extend_bbox((x, y, w, h))
                faces.append((x, y, w, h))

    return faces


def face_detect(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(img)
    result = []

    for face in face_locations:
        top, right, bottom, left = face

        x, y, w, h = (left, top, right - left, bottom - top)

        if h > face_min_height:
            x, y, w, h = extend_bbox(((x, y, w, h)))
            result.append((x, y, w, h))

    return result


count = 0
frame_face_count = 0

pbar = tqdm(total=length)

faces = []
faces_attr = []

while (cap.isOpened()):
    ret, frame = cap.read()

    if ret:
        faces = face_detect_mtcnn(frame)
        frame = show_faces(frame, faces)
        out.write(frame)

        pbar.update()

    else:
        break

cap.release()
out.release()
pbar.close()
