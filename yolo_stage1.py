from ultralytics import YOLO

model = YOLO("/content/ocr_test/yolo11x.pt")

results = model.train(
    data="/content/ocr_test/segmentation_model/yolo_stage1.yaml",
    epochs=100,
    augment=True,
    imgsz=640,
    batch=20,
    device="cuda"
)

predictions = model("/content/ocr_test/dataset_small/images/passport_000000.png")
