from ultralytics import YOLO

model = YOLO("yolo11x.pt")

results = model.train(
    data="/content/ocr_test/segmentation_model/yolo_stage2.yaml",
    epochs=100,
    imgsz=640,
    batch=14,
    device="cuda"
)

predictions = model("path/to/test_image.jpg")
