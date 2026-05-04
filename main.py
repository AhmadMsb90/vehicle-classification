import argparse

from src.models.classifier import get_model
from src.inference.video_processor import VideoProcessor
from src.config import MODEL_PATH


def main(video_path):
    model = get_model(pretrained=False)
    processor = VideoProcessor(model, MODEL_PATH)
    processor.process(video_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    args = parser.parse_args()

    main(args.video)