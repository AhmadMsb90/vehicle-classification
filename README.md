# Vehicle Detection and Classification Pipeline

This project implements a two-stage pipeline for vehicle detection and classification in a video. It combines YOLOv8 for object detection and ResNet18-based classifiers for vehicle classification.

The system processes video frames, detects vehicles, extracts bounding boxes, and classifies each detected region. A two-level classification setup is supported:
- Level 1: light vs heavy vehicle classification
- Level 2: fine-grained classification of light vehicle models

## Project Structure

```
vehicle-classification/
├── assets/
│   ├── heavy_vehicle.png
│   └── light_vehicle.png
├── data/
│   ├── dataset_level1/
│   │   ├── train/
│   │   └── val/
│   ├── dataset_level2/
│   │   ├── train/
│   │   └── val/
│   └── videos/
├── models/
│   ├── light_heavy_model.pth
│   ├── level2_vehicle_model.pth
│   └── yolov8n.pt
├── scripts/
│   ├── train.py
│   └── run_video.py
├── src/
│   ├── config.py
│   ├── data/
│   │   └── dataloader.py
│   ├── inference/
│   │   └── video_processor.py
│   ├── models/
│   │   ├── classifier.py
│   │   └── detector.py
│   ├── training/
│   │   └── trainer.py
│   └── utils/
```

## Models

YOLOv8 is used for object detection. It detects vehicles and outputs bounding boxes for each frame.

A ResNet18 model is used for classification with two stages:

- Level 1 model: binary classification (light / heavy)
- Level 2 model: multi-class classification of light vehicle models

Only the final fully connected layer is trained; the backbone is frozen during training.

## Dataset

### Dataset Description

The dataset used in this project was collected from real traffic at a tollgate in Iran. A Sony SR46 DCR Handycam was mounted on a tripod at a height of approximately 10–15 meters, positioned to provide a stable front-view perspective of vehicles passing through the lane.

The videos were recorded in .MPG format at a resolution of 720×576 and a frame rate of 25 FPS, resulting in approximately 2 hours and 15 minutes of footage.

Recording sessions were conducted under varying environmental conditions, including cloudy, semi-sunny, and sunny weather. This introduced natural variations in lighting, shadows, and visibility.

Training images were extracted from these videos by selecting frames when a vehicle fully entered a predefined region of interest. The corresponding cropped vehicle regions were then used as samples for model training.

The dataset follows a PyTorch ImageFolder structure.

### Level 1 dataset

The Level 1 dataset is designed for binary classification of vehicles into two categories: light and heavy. It contains cropped vehicle images extracted from traffic video frames and organized using the standard train/validation split. The "light" class includes smaller passenger vehicles, while the "heavy" class includes larger vehicles such as trucks and buses. This dataset is used to train the first-stage classifier, which determines whether a detected vehicle should undergo further classification.

```
dataset_level1/
├── train/
│   ├── light/
│   └── heavy/
└── val/
    ├── light/
    └── heavy/
```

### Level 2 dataset

The Level 2 dataset is used for fine-grained classification of light vehicles into specific models. It contains cropped images of light vehicles only, organized into multiple classes corresponding to individual vehicle types common in Iran, including Mazda_2000, Nissan_Zamiad, various Peugeot models, Peykan, Pride variants, Quik, Renault_L90, Samand, and Tiba2. Each class represents a distinct vehicle model commonly observed in the collected traffic data. This dataset is used to train the second-stage classifier, which is applied only to vehicles classified as "light" in the first stage.

```
dataset_level2/
├── train/
│   ├── Mazda_2000
│   ├── Nissan_Zamiad
│   ├── Peugeot_206
│   ├── Peugeot_207
│   ├── Peugeot_405
│   ├── Peugeot_Pars
│   ├── Peykan
│   ├── Pride_111
│   ├── Pride-131
│   ├── Quik
│   ├── Renault_L90
│   ├── Samand
│   └── Tiba2
└── val/
    ├── Mazda_2000
    ├── Nissan_Zamiad
    ├── Peugeot_206
    ├── Peugeot_207
    ├── Peugeot_405
    ├── Peugeot_Pars
    ├── Peykan
    ├── Pride_111
    ├── Pride-131
    ├── Quik
    ├── Renault_L90
    ├── Samand
    └── Tiba2
```


### Dataset Statistics

The data is divided into two classification levels. Each class is split into training (80%) and validation (20%) sets.

#### Level 1: Category Classification
This level distinguishes between light and heavy vehicle types for high-level traffic analysis.


| Category | Total | Training | Validation |
| :--- | :--- | :--- | :--- |
| Light | 10,204 | 8,163 | 2,041 |
| Heavy | 1,269 | 1,015 | 254 |
| **Total** | **11,473** | **9,178** | **2,295** |

#### Level 2: Model Classification
This level identifies specific vehicle models across 13 different classes.


| Model | Total | Training | Validation |
| :--- | :--- | :--- | :--- |
| Quik | 1,005 | 804 | 201 |
| Pride 111 | 858 | 686 | 172 |
| Samand | 849 | 679 | 170 |
| Tiba 2 | 832 | 665 | 167 |
| Peugeot 206 | 791 | 632 | 159 |
| Peykan | 770 | 616 | 154 |
| Pride 131 | 749 | 599 | 150 |
| Peugeot Pars | 744 | 595 | 149 |
| Peugeot 207 | 737 | 589 | 148 |
| Renault L90 | 648 | 518 | 130 |
| Mazda 2000 | 607 | 485 | 122 |
| Nissan Zamiad | 597 | 477 | 120 |
| Peugeot 405 | 592 | 473 | 119 |
| **Total** | **8,829** | **7,063** | **1,766** |


Images are resized to 224×224 and normalized using ImageNet statistics.

## Training

Training is executed using:

### Level 1 training
```
python -m scripts.train --level level1
```

### Level 2 training
```
python -m scripts.train --level level2
```

### Training pipeline

- Load dataset using PyTorch DataLoader
- Initialize pretrained ResNet18
- Replace final classification layer
- Freeze backbone parameters
- Train classifier head only
- Evaluate on validation set


### Training Results

#### Level 1 (Light vs Heavy)

The Level 1 classifier converges quickly and achieves strong performance, reaching 97% validation accuracy. This indicates that the model can reliably distinguish between light and heavy vehicles under the given dataset conditions.

#### Level 2 (Light Vehicle Models)

The Level 2 classifier performs fine-grained classification across 13 light vehicle models. The model reaches 70% validation accuracy, showing steady learning progress while reflecting the higher difficulty of distinguishing between visually similar vehicle classes.






Trained models are saved inside the `models/` directory.

## Inference

### Level 1 inference
```
python -m scripts.run_video --mode level1 --video data/videos/<video_name>.mp4
```

### Level 2 inference
```
python -m scripts.run_video --mode level2 --video data/videos/<video_name>.mp4
```

### Inference pipeline

- Reads video frame by frame
- Runs YOLOv8 object detection
- Filters vehicle bounding boxes
- Extracts region of interest (ROI)
- Applies Level 1 classification (light vs heavy)
- Applies Level 2 classification only if the vehicle is classified as light
- Draws bounding boxes and labels
- Displays processed video in real time

Press `q` to exit the video window.

## Requirements

Install dependencies:

```
pip install -r requirements.txt
```

### Main dependencies

- torch
- torchvision
- ultralytics
- opencv-python
- pillow
- tqdm

## Sample Results

<p align="center">
  <img src="assets/light_vehicle.png" width="45%" />
  <img src="assets/heavy_vehicle.png" width="45%" />
</p>