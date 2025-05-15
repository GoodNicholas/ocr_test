from ultralytics import YOLO

model = YOLO("yolov11x.pt")

results = model.train(
    data="/home/nicholas/PycharmProjects/Passport_generator/segmentation_model/yolo_stage1.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device="cpu"
)

predictions = model("path/to/test_image.jpg")
