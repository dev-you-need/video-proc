import cv2
import numpy as np
from typing import Tuple

from video_proc.person import Person


def put_text(img: np.ndarray,
             bbox: Tuple[float, float, float, float],
             person: Person,
             all_frames_cnt: int) -> np.ndarray:
    """
    Function for putting information about a person on video
    
    Parameters
    ----------
    img: np.ndarray
        All frame on each person is located
    bbox: Tuple[float, float, float, float]
        BBox where person is located in x_top, y_top, weight, height format
    person:Person
        Class that identified person on video
    all_frames_cnt: int
        Amount of all frames in video. Using for calculating percentage of person appearing in frames

    Returns
    -------
    np.ndarray
        Frame with additional info about a person (bbox, gender, age, race)

    """
    x, y, w, h = bbox

    pivot_img_size = 112
    resolution_x = img.shape[1];

    overlay = img.copy()
    opacity = 0.4

    img = cv2.rectangle(img, (x, y), (x + w, y + h), (67, 67, 67), 1)
    if x + w + pivot_img_size < resolution_x:
        # right
        cv2.rectangle(img, (x + w, y), (x + w + pivot_img_size + 20, y + h), (64, 64, 64), cv2.FILLED)
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    elif x - pivot_img_size > 0:
        # left
        cv2.rectangle(img, (x - pivot_img_size - 20, y), (x, y + h), (64, 64, 64), cv2.FILLED)
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    gender_label = person.gender
    race_label = person.race
    age_label = person.age
    num_frames = person.percent_in_video(all_frames_cnt)

    attributes = [gender_label, race_label, age_label, num_frames]

    for index, attribute in enumerate(attributes):
        bar_x = 35  # this is the size if an emotion is 100%

        if x + w + pivot_img_size < resolution_x:

            text_location_y = y + 20 + (index + 1) * 20
            text_location_x = x + w

            if text_location_y < y + h:
                cv2.putText(img, attribute, (text_location_x + 10, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

        elif x - pivot_img_size > 0:

            text_location_y = y + 20 + (index + 1) * 20
            text_location_x = x - pivot_img_size

            if text_location_y <= y + h:
                cv2.putText(img, attribute, (text_location_x + 10, text_location_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

    return img
