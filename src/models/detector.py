from ultralytics import YOLO
from src.config import YOLO_MODEL

def load_detector():
    return YOLO(YOLO_MODEL)