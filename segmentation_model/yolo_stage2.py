from ultralytics import YOLO

model = YOLO("/content/drive/MyDrive/Работа/yolo11x.pt")

results = model.train(
    data="/content/ocr_test/segmentation_model/yolo_stage2.yaml",
    epochs=100,
    imgsz=640,
    augment=True,
    batch=40,
    device="cuda",
    pretrained=False
)

predictions = model("/content/ocr_test/dataset_small/images/passport_000000.png")
