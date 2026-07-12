from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# Repo folders:
#   data/dataset/obj_train_data/   <- training images + matching .txt labels
#   data/dataset/obj.names         <- class names from CVAT YOLO export
#   models/trained/weights/best.pt <- trained model output

DATASET_DIR = ROOT / "data" / "dataset"
IMAGES_AND_LABELS = "obj_train_data"
NAMES_FILE = DATASET_DIR / "obj.names"

MODEL_DIR = ROOT / "models" / "trained"
BEST_WEIGHTS = MODEL_DIR / "weights" / "best.pt"

PRETRAINED = "yolov8n.pt"
EPOCHS = 50
IMG_SIZE = 640

class_names = [
    line.strip()
    for line in NAMES_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if not class_names:
    raise SystemExit(f"No class names found in {NAMES_FILE}")

names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
data_yaml = DATASET_DIR / "dataset.yaml"
data_yaml.write_text(
    f"path: {DATASET_DIR.as_posix()}\n"
    f"train: {IMAGES_AND_LABELS}\n"
    f"val: {IMAGES_AND_LABELS}\n"
    f"nc: {len(class_names)}\n"
    f"names:\n"
    f"{names_block}\n",
    encoding="utf-8",
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
