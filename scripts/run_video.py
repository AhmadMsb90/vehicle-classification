import argparse
import torch
from ultralytics import YOLO
from torchvision import models
import torch.nn as nn

from src.config import *
from src.inference.video_processor import VideoProcessor
from torchvision.models import resnet18

# loads yolo detector and classification models for level1 and optionally level2
def load_models(level2=False):

    # load yolo detector
    detector = YOLO(YOLO_WEIGHTS)

    # load level1 classifier (heavy vs light)
    model1 = resnet18(weights=None)
    model1.fc = nn.Linear(model1.fc.in_features, 2)
    model1.load_state_dict(torch.load(LEVEL1_WEIGHTS, map_location=DEVICE))

    # initialize level2 classifier
    model2 = None
    if level2:
        # load level2 classifier (fine-grained classes)
        model2 = resnet18(weights=None)
        model2.fc = nn.Linear(model2.fc.in_features, len(CLASS_NAMES_LEVEL2))
        model2.load_state_dict(torch.load(LEVEL2_WEIGHTS, map_location=DEVICE))

    # return detector and both classifiers
    return detector, model1, model2


def main(args):

    # load detector and classifiers depending on mode
    detector, model1, model2 = load_models(args.mode == "level2")

    # create video processor with loaded models
    processor = VideoProcessor(
        detector=detector,
        model1=model1,
        model2=model2,
        class_names=CLASS_NAMES_LEVEL2,
        device=DEVICE
    )

    # run video processing
    processor.run(args.video, mode=args.mode)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # add mode argument (level1 or level2)
    parser.add_argument("--mode", choices=["level1", "level2"], required=True)

    # add video path argument
    parser.add_argument("--video", required=True)

    # parse command-line arguments
    args = parser.parse_args()

    main(args)
