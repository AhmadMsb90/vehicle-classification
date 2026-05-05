from ultralytics import YOLO
from src.config import YOLO_WEIGHTS

# loads and returns the yolo detector using configured weights
def load_detector():
    # create yolo model instance with pretrained weights
    return YOLO(YOLO_WEIGHTS)
