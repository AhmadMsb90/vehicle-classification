import cv2
import torch
from PIL import Image
from torchvision import transforms

class VideoProcessor:
    # handles video processing, detection, and two-stage classification

    def __init__(self, detector, model1, model2=None, class_names=None, device="cpu"):
        # constructor initializes detector, models, class names, device, and transforms
        self.detector = detector
        self.model1 = model1.to(device).eval()
        self.model2 = model2.to(device).eval() if model2 else None
        self.class_names = class_names
        self.device = device

        # define preprocessing pipeline for roi images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def run(self, video_path, mode="level2"):
        
        # runs video loop: detect => crop => classify => draw => display
        
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            # read next frame
            ret, frame = cap.read()
            if not ret:
                break

            # resize frame for detector
            frame = cv2.resize(frame, (640, 640))
            # run object detector
            results = self.detector(frame)

            for r in results:
                for box in r.boxes:

                    # extract bounding box coordinates
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    # crop region of interest
                    roi = frame[y1:y2, x1:x2]

                    # skip empty roi
                    if roi.size == 0:
                        continue

                    # convert roi to pil and preprocess
                    img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    img = self.transform(img).unsqueeze(0).to(self.device)

                    # run first-level classifier
                    with torch.no_grad():
                        out1 = self.model1(img)
                        label1 = out1.argmax(1).item()

                    # map prediction to label text
                    label_text = "heavy" if label1 == 0 else "light"

                    # run second-level classifier if needed
                    if mode == "level2" and label_text == "light":
                        out2 = self.model2(img)
                        label2 = self.class_names[out2.argmax(1).item()]
                        label_text = f"{label_text} | {label2}"

                    # draw bounding box
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    # draw label text
                    cv2.putText(frame, label_text, (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # show processed frame
            cv2.imshow("video", frame)

            # exit on q key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # release video and close windows
        cap.release()
        cv2.destroyAllWindows()
