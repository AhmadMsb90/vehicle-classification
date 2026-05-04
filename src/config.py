import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data/dataset_level1"
MODEL_PATH = "models/light_heavy_model.pth"
YOLO_MODEL = "models/yolov8n.pt"
VIDEO_PATH = "data/videos/M2U00107.mp4"

BATCH_SIZE = 8
EPOCHS = 5
LR = 0.001
NUM_CLASSES = 2