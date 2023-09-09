import cv2
import numpy as np

def cv2_to_array(frame):
    '''
    frame is in original size
    '''
    frame = cv2.resize(frame, dsize=[28,28])
    frame = 255-frame
    frame = cv2.threshold(frame, 170, 255, cv2.THRESH_TOZERO)[1]
    
    return np.asarray(frame)