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

    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    img = cv2.putText(img, person.age, (int(x), int(y + h)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    img = cv2.putText(img, person.gender, (int(x), int(y + h+22)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    img = cv2.putText(img, person.age, (int(x), int(y + h + 44)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    img = cv2.putText(img, person.percent_in_video(all_frames_cnt), (int(x), int(y + h + 66)),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return img
