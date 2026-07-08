from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# Repo folders:
#   data/dataset/obj_train_data/   <- training images + matching .txt labels
#   models/trained/weights/best.pt <- trained model output

DATASET_DIR = ROOT / "data" / "dataset"
IMAGES_AND_LABELS = "obj_train_data"

MODEL_DIR = ROOT / "models" / "trained"
BEST_WEIGHTS = MODEL_DIR / "weights" / "best.pt"

PRETRAINED = "yolov8n.pt"
EPOCHS = 50
IMG_SIZE = 640

data_yaml = DATASET_DIR / "dataset.yaml"
data_yaml.write_text(
    f"path: {DATASET_DIR.as_posix()}\n"
    f"train: {IMAGES_AND_LABELS}\n"
    f"val: {IMAGES_AND_LABELS}\n"
    f"nc: 1\n"
    f"names:\n"
    f"  0: fish\n"
)

model = YOLO(PRETRAINED)
model.train(
    data=str(data_yaml),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    project=str(MODEL_DIR.parent),
    name=MODEL_DIR.name,
)

print(f"best.pt -> {BEST_WEIGHTS}")
