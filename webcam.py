from ultralytics import YOLO
import cv2
import math 

# Code eexamples and snippets gotten from -- https://datalab.medium.com/yolov8-detection-from-webcam-step-by-step-cpu-d590a0700e36
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO('./runs/detect/yolov8n_chess_3/weights/best.pt')

classNames = [
    'black_bishop',
    'black_king',
    'black_knight',
    'black_pawn',
    'black_queen',
    'black_rook',
    'white_bishop',
    'white_king',
    'white_knight',
    'white_pawn',
    'white_queen',
    'white_rook'
]



while True:
    success, img = cap.read()
    results = model(img, stream=True, conf=0.5)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) 
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
            confidence = math.ceil((box.conf[0]*100))/100
            print(confidence)
            cls = int(box.cls[0])
            print(classNames[cls])
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (0, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()