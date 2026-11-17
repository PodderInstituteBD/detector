import cv2
import json
import os

# 1. Load labels if available, or define a fallback dictionary
# (Make sure 'labels.json' is in the same folder if you use a file)
labels_file = "labels.json"
if os.path.exists(labels_file):
    with open(labels_file, "r") as f:
        labels = json.load(f)
else:
    # Default placeholder classes if labels.json isn't created yet
    labels = {0: "object", 1: "person", 2: "cell phone"}

def process_detections(frame, detections, confidence_threshold=0.5):
    """
    Scores the model's guesses, filters by confidence, 
    picks a name, and draws sure boxes on the camera frame.
    """
    height, width, _ = frame.shape
    
    for det in detections:
        score = det.get("score", 0.0)
        
        # Keep only the boxes that meet or exceed our confidence threshold
        if score >= confidence_threshold:
            class_id = det.get("class_id", 0)
            label_name = labels.get(str(class_id), f"Class {class_id}")
            
            # Get bounding box coordinates
            box = det.get("box", [100, 100, 200, 200])
            ymin, xmin, ymax, xmax = int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)
            
            # Draw the bounding box on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Draw the label and confidence score
            display_text = f"{label_name}: {score:.2f}"
            cv2.putText(frame, display_text, (xmin, max(35, ymin - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    return frame

def main():
    # Initialize the camera (0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting live object detector. Press 's' to take a screenshot, or 'q' to quit.")

    confidence_threshold = 0.5  # Change this value to adjust how sure it must be (e.g., 0.7 for stricter checks)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Simulated model guesses for demonstration 
        # (In your actual assignment model pipeline, replace this with your model's real output array)
        simulated_detections = [
            {"class_id": 1, "score": 0.85, "box": [0.2, 0.2, 0.7, 0.6]}
        ]

        # Process and draw sure boxes onto the camera frame
        processed_frame = process_detections(frame, simulated_detections, confidence_threshold)

        # Show the live feed on your screen
        cv2.imshow("Live Object Detector - Assignment", processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Take a screenshot and save it as an image file
            screenshot_path = "assignment_screenshot.png"
            cv2.imwrite(screenshot_path, processed_frame)
            print(f"Screenshot saved successfully as '{screenshot_path}'! You can now show it to your parent.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
