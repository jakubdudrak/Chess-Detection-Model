from ultralytics import YOLO

model = YOLO('./runs/detect/yolov8n_chess_42/weights/best.pt')

results = model('data/images/IMG_0717.mov', save=True, conf=0.5)

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs