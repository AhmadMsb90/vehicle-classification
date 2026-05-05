import argparse
import torch

from src.config import DEVICE, LEVEL1_DATA, LEVEL2_DATA, LEVEL1_WEIGHTS, LEVEL2_WEIGHTS
from src.data.dataloader import get_dataloaders
from src.models.classifier import build_resnet
from src.training.trainer import train_model

def main(args):

    # choose dataset and save path based on level
    if args.level == "level1":
        data_dir = LEVEL1_DATA
        save_path = LEVEL1_WEIGHTS
        num_classes = 2

    elif args.level == "level2":
        data_dir = LEVEL2_DATA
        save_path = LEVEL2_WEIGHTS
        # load dataloaders to get class names for level2
        train_loader, val_loader, class_names = get_dataloaders(data_dir)
        num_classes = len(class_names)

    else:
        # invalid level argument
        raise ValueError("level must be level1 or level2")

    # load train and validation dataloaders
    train_loader, val_loader, _ = get_dataloaders(data_dir)

    # build resnet model with correct number of classes
    model = build_resnet(num_classes)

    # train model with given settings
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        epochs=5,
        save_path=save_path
    )


if __name__ == "__main__":
    # create argument parser
    parser = argparse.ArgumentParser()

    # add level argument (level1 or level2)
    parser.add_argument("--level", type=str, required=True, choices=["level1", "level2"])

    # parse command-line arguments
    args = parser.parse_args()

    # run main function
    main(args)
