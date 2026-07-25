from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="D:/My First Project.v11i.yolov8/data.yaml",
    epochs=25,
    imgsz=640,
    batch=16,
    project="D:/Dataset/Currency_detection/runs",  # folder where runs will be saved
    name="currency_model"  # experiment name
)