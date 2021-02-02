import numpy as np
from deepface.extendedmodels import Race
from deepface.extendedmodels import Gender
from deepface.extendedmodels import Age
from arcface import ArcFace
import mtcnn

from video_proc.image_preprocessing import img_preprocess

face_rec = ArcFace.ArcFace()
race_model = Race.loadModel()
gender_model = Gender.loadModel()
age_model = Age.loadModel()
face_detector = mtcnn.MTCNN()


class Person(object):

    def __init__(self, img: np.ndarray):
        """
        Class for identifying person in Video

        Parameters
        ----------
        img: np.ndarray
        """

        self.age_predictions = []
        self.race_predictions = []
        self.gender_predictions = []
        self.img = img  # TODO Delete this variable. Use just to check anchor image
        self.add_picture(img)

        self.person_encoding = face_rec.calc_emb(img)

    def add_picture(self, img: np.ndarray):
        """
        Method for adding information about person from new picture

        Parameters
        ----------
        img: np.ndarray

        """

        img_pixels = img_preprocess(img)

        # add age
        preds = age_model.predict(img_pixels)[0, :]
        self.age_predictions.append(preds)

        # add race
        preds = race_model.predict(img_pixels)
        self.race_predictions.append(preds)

        # add gender
        gender_prediction = gender_model.predict(img_pixels)[0, :]
        self.gender_predictions.append(gender_prediction)

    @property
    def race(self) -> str:
        """
        Property for generating text with race info about person

        Returns
        -------
        str
            Information about person race and probability of this race

        """
        race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino']
        preds = sum(self.race_predictions) / len(self.race_predictions)
        return "{}: {:02.0f}%".format(race_labels[np.argmax(preds)], np.max(preds) * 100)

    @property
    def age(self) -> str:
        """
        Property for generating text with age info about person

        Returns
        -------
        str
            Information about person age and probability of this age
        """

        preds = sum(self.age_predictions) / len(self.age_predictions)
        return "Age: {}".format(int(Age.findApparentAge(preds)))

    @property
    def gender(self) -> str:
        """
        Property for generating text with gender info about person

        Returns
        -------
        str
            Information about person gender and probability of this gender
        """

        preds = sum(self.gender_predictions) / len(self.gender_predictions)

        if np.argmax(preds) == 0:
            gender = "Woman"
        elif np.argmax(preds) == 1:
            gender = "Man"

        return '{}: {:02.0f}%'.format(gender, np.max(preds) * 100)

    def percent_in_video(self, all_frames_cnt: int) -> str:
        """
        Method for calculating percentage of person appearing in frames

        Parameters
        ----------
        all_frames_cnt: int
            Amount of all frames in video.

        Returns
        -------
        str
            Text information about percentage of appearing person in video
        """

        return "" # f'Percentage of frames {round(len(self.age_predictions) / all_frames_cnt, 3)}'

    def verify_person(self, img: np.ndarray) -> bool:
        """
        Method for verifying person

        Parameters
        ----------
        img: np.ndarray
            Picture of new person

        Returns
        -------
        bool
            True - if person are the same, False - otherwise

        """
        person_encoding = face_rec.calc_emb(img)
        if face_rec.get_distance_embeddings(self.person_encoding, person_encoding) < 0.687:
            return True
        return False
