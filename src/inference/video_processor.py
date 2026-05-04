import cv2
import torch
from PIL import Image
from torchvision import transforms

from src.config import DEVICE
from src.models.detector import load_detector


class VideoProcessor:
    def __init__(self, classifier, classifier_weights):
        self.detector = load_detector()

        self.classifier = classifier.to(DEVICE)
        self.classifier.load_state_dict(torch.load(classifier_weights, map_location=DEVICE))
        self.classifier.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def process(self, video_path):
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, (640, 640))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            results = self.detector(frame_rgb)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    roi = frame_resized[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue

                    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    img = self.transform(img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        outputs = self.classifier(img)
                        _, pred = torch.max(outputs, 1)

                    label = "light" if pred.item() == 1 else "heavy"

                    cv2.putText(frame_resized, label, (x2 - 80, y2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Video", frame_resized)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()