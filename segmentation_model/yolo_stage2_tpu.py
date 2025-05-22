import torch
import torch_xla.core.xla_model as xm
from ultralytics import YOLO

device = xm.xla_device()

model = YOLO("/content/ocr_test/yolo11x.pt")

results = model.train(
    data="/content/ocr_test/segmentation_model/yolo_stage2.yaml",
    epochs=100,
    imgsz=640,
    augment=True,
    batch=60,
    device=device
)

predictions = model("/content/ocr_test/dataset_small/images/passport_000000.png")
