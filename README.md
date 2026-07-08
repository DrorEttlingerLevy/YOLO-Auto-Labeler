# YOLO Auto-Labeler

Semi-automatic YOLO dataset labeling tool. A trained model proposes bounding boxes; you review, fix, and save labels.

## Folder layout

```
data/
  dataset/obj_train_data/   # training images + YOLO .txt labels
  images_to_label/          # new images for the labeler
  labels/                   # labels saved by the labeler
models/
  trained/weights/best.pt   # output from basic_YOLO_train.py
```

## Requirements

- Python 3.8+
- A **trained YOLOv8 model** (`best.pt` from an Ultralytics training run) that detects your target classes reasonably well
- Images to label (`.jpg`, `.jpeg`, `.png`)

```bash
pip install ultralytics opencv-python pillow
```

(`tkinter` is included with standard Python on Windows.)

## Setup

Edit the paths at the top of `auto.py`:

- `images_dir` — folder with images to label
- `labels_dir` — folder where YOLO `.txt` labels are saved
- `model_path` — path to your trained `best.pt` weights

## How it works

1. Loads your YOLO model and runs inference on each unlabeled image.
2. Opens a GUI showing detections (red = accepted, gray = rejected).
3. You accept/reject boxes by clicking them or using the thumbnail panel.
4. Press **Add Box** (`n`) to draw missing boxes manually.
5. Press **Save & Next** (`m`) to write a YOLO-format label file and move on.

Label format: `class x_center y_center width height` (normalized 0–1).

Images that already have a matching `.txt` file in `labels_dir` are skipped.

## Run

```bash
python auto.py
```
