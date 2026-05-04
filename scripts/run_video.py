from src.models.classifier import get_model
from src.inference.video_processor import VideoProcessor
from src.config import MODEL_PATH, VIDEO_PATH

model = get_model(pretrained=False)

processor = VideoProcessor(model, MODEL_PATH)
processor.process(VIDEO_PATH)