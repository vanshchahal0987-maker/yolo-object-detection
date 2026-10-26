from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Read the test image
image_path = "test_fixed.jpg"
print("Image exists:", cv2.haveImageReader(image_path))
# Detect objects
results = model(image_path)

# Display and save the result
for result in results:
    output_image = result.plot()

    cv2.imshow("Object Detection", output_image)
    cv2.imwrite("detected_output.jpg", output_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Object detection completed!")
print("Result saved as detected_output.jpg")