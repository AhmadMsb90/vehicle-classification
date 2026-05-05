import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paths
DATA_DIR = "data"
MODEL_DIR = "models"

LEVEL1_DATA = f"{DATA_DIR}/dataset_level1"
LEVEL2_DATA = f"{DATA_DIR}/dataset_level2"

YOLO_WEIGHTS = f"{MODEL_DIR}/yolov8n.pt"
LEVEL1_WEIGHTS = f"{MODEL_DIR}/light_heavy_model.pth"
LEVEL2_WEIGHTS = f"{MODEL_DIR}/level2_vehicle_model.pth"

# training
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 5

CLASS_NAMES_LEVEL2 = [
    'Mazda_2000', 'Nissan_Zamiad', 'Peugeot_206', 'Peugeot_207',
    'Peugeot_405', 'Peugeot_Pars', 'Peykan', 'Pride-131',
    'Pride_111', 'Quik', 'Renault_L90', 'Samand', 'Tiba2'
]