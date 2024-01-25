from ultralytics import YOLO

model = YOLO('yolov8m.pt')

# Training.
results = model.train(
   data='data.yaml',
   imgsz=640,
   epochs=50,
   batch=8,
   name='yolov8n_chess_4')