import cv2
import numpy as np


def get_merge_frame(color_path, event_path):
    rgb = cv2.imread(color_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    ev = cv2.imread(event_path)
    ev = cv2.cvtColor(ev, cv2.COLOR_BGR2RGB)
    img = cv2.merge((rgb, ev))
    return img
