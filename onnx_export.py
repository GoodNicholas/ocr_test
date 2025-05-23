from ultralytics import YOLO

model = YOLO('/content/ocr_test/runs/detect/train/weights/best.pt')
model.export(format='onnx', nms=True, simplify=True)
