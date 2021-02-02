import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from typing import Tuple


def img_preprocess(img: np.ndarray,
                   target_size: Tuple[float, float] = (224, 224)
                   ) -> np.ndarray:
    """
    Function for preprocessing images before Deepface model

    Parameters
    ----------
    img: np.ndarray
        Image array
    target_size: Tuple[float, float]
        Size needed for model

    Returns
    -------
    np.ndarray
        Preprocessed picture

    """
    img = cv2.resize(img, target_size)
    img_pixels = image.img_to_array(img)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255

    return img_pixels
