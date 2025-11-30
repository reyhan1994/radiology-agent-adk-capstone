# utils/preprocessing.py
from PIL import Image


def load_image(path):
img = Image.open(path)
return img
