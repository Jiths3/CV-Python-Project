import cv2

classNames = { 0: 'background',
1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
14: 'motorbike', 15: 'person', 16: 'pottedplant',
17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

proto = 'MobileNetSSD_deploy.prototxt'
weights = 'MobileNetSSD_deploy.caffemodel'
model = cv2.dnn.readNetFromCaffe(proto,weights)

inputSize = (320, 320)
inputScale = 1/125

# Create a VideoCapture object to connect to the camera
cap = cv2.VideoCapture(0)

# Set the resolution of the captured frame
cap.set(3, 640) # Width
cap.set(4, 480) # Height

while True:
    # Read the current frame from the camera
    success, img = cap.read()
    # Preprocess the image
    blob = cv2.dnn.blobFromImage(img, inputScale, inputSize, mean=(127.5, 127.5, 127.5),  swapRB=True, crop=False)
    model.setInput(blob)
    detections = model.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])
        # Check if the detection confidence is above a threshold
        if confidence > 0.5: 
            # Get the bounding box coordinates
            box = detections[0, 0, i, 3:7] * [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
            (startX, startY, endX, endY) = box.astype(int)
            # Draw the bounding box and label on the image
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f'{classNames[class_id]}: {confidence:.2f}'
            cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Object Detection', img)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
