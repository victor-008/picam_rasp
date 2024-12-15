# requisite packages
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import csv
from datetime import datetime

# Setting up the camera with Picam
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1280, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Loading the model
model = YOLO("/home/Rono/FYP/best.pt")

# Path to the CSV file
csv_file_path = "/home/Rono/FYP/detections.csv"

# Open the CSV file in write mode
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header
    writer.writerow(['Frame', 'Timestamp', 'Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
    
    frame_count = 0
    
    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()
        
        # Run the model on the captured frame and store the results
        results = model(frame)
        
        # Output the visual detection data, drawn on the camera preview window
        annotated_frame = results[0].plot()
        
        # Get inference time
        inference_time = results[0].speed['inference']
        fps = 1000 / inference_time  # Converts to milliseconds
        text = f'FPS: {fps:.1f}'
        
        # Define font and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = annotated_frame.shape[1] - text_size[0] - 16  # 10 pixels from the right
        text_y = text_size[1] + 10  # 10 pixels from the top

        # Draw the text on the annotated frame
        cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow("Camera feed", annotated_frame)
        
        # Write detections to the CSV file
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            for detection in results[0].boxes:
                class_index = int(detection.cls.item())  # Convert tensor to int
                class_name = model.names[class_index]  # Get class name from index
                confidence = float(detection.conf.item())  # Convert tensor to float
                x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())  # Convert tensor to list and then to int
                writer.writerow([frame_count, timestamp, class_name, confidence, x1, y1, x2, y2])

        frame_count += 1

        # press "q" to exit
        if cv2.waitKey(1) == ord("q"):
            break

# Close all windows
cv2.destroyAllWindows()