import cv2
import numpy as np
import time

model_path = "yolov3.weights"
config_path = "yolov3.cfg"
labels_path = "coco.names"

with open(labels_path, 'r') as f:
    class_labels = f.read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(config_path, model_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
except AttributeError:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

reference_object_label = "bottle"  
reference_object_width_cm = 7.0  

pixel_to_cm_ratio = None

cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object Detection', 1280, 720)

while True:
    start_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    frame = cv2.flip(frame, 1)

    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                (centerX, centerY, w, h) = box.astype("int")
                startX = int(centerX - (w / 2))
                startY = int(centerY - (h / 2))

                boxes.append([startX, startY, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            (startX, startY) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            endX = startX + w
            endY = startY + h

            label = str(class_labels[class_ids[i]])
            confidence = confidences[i]

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if label == reference_object_label and pixel_to_cm_ratio is None:
                pixel_to_cm_ratio = reference_object_width_cm / w

            if pixel_to_cm_ratio:
                width_cm = w * pixel_to_cm_ratio
                height_cm = h * pixel_to_cm_ratio

                dimension_text = f"W: {width_cm:.2f} cm, H: {height_cm:.2f} cm"
            else:
                width_cm = w * (reference_object_width_cm / w)
                height_cm = h * (reference_object_width_cm / w)
                dimension_text = f"W: {width_cm:.2f} cm, H: {height_cm:.2f} cm"
            
            cv2.putText(frame, dimension_text, (startX, startY + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
