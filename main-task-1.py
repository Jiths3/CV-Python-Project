import cv2
import os
from datetime import datetime

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create a VideoCapture object to connect to the camera
camera = cv2.VideoCapture(0)

# Set the resolution of the captured frame
camera.set(3, 640)  # Width
camera.set(4, 480)  # Height

# Create a background subtractor object
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Variables to track if motion is detected and image is captured
motion_detected = False
image_captured = False

while True:
    # Read the current frame from the camera
    ret, frame = camera.read()

    if not ret:
        break

    # Apply the background subtraction to detect moving objects
    fg_mask = background_subtractor.apply(frame)

    # Apply thresholding to remove shadows
    _, thresh = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)

    # Find contours of the moving objects
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Ignore small contours (noise)
        if cv2.contourArea(contour) < 500:
            continue

        # Calculate the area of the contour and the total frame area
        contour_area = cv2.contourArea(contour)
        frame_area = frame.shape[0] * frame.shape[1]

        # Calculate the percentage of motion in the frame
        motion_percentage = (contour_area / frame_area) * 100

        # Check if the motion percentage exceeds the threshold (e.g., 50%)
        if motion_percentage >= 10 and not image_captured:
            # Draw a bounding rectangle around the moving object
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Set motion_detected and image_captured variables to True
            motion_detected = True
            image_captured = True

    # Display the resulting frame
    cv2.imshow('Motion Detection', frame)

    # Capture a photo when motion is detected
    if motion_detected and image_captured:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S.%f')[:-3]  # Include milliseconds
        image_filename = f'motion_detected_{timestamp}.jpg'
        image_path = os.path.join(current_dir, image_filename)
        cv2.imwrite(image_path, frame)
        print('Image saved:', image_path)
        motion_detected = False

    # Reset image_captured variable after a certain delay
    if image_captured and cv2.waitKey(250) == -1:
        image_captured = False

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
