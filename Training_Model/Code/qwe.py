# Load YOLOv8 model - using yolov8s.pt (small model, balanced between speed and accuracy)
model = YOLO("yolov8s.pt")

# Train the model
model.train(
    data="/content/YOLOv8_Project/data.yaml",  # <-- Replace with your actual path
    epochs=100,                           # Number of training epochs
    imgsz=640,                            # Input image size
    batch=16,                             # Batch size (adjust based on GPU memory)
    name="yolov8s_phone_detect",          # Experiment name

    # Optimization settings
    optimizer="AdamW",                    # Optimizer (AdamW for better generalization)
    lr0=2e-3,                             # Initial learning rate
    lrf=0.01,                             # Final learning rate (as a fraction of lr0)
    weight_decay=0.001,                   # Regularization to prevent overfitting
    momentum=0.937,                       # Not used with AdamW, but kept for consistency
    patience=15,                          # Early stopping patience

    # Warmup settings
    warmup_epochs=3.0,                    # Number of warmup epochs
    warmup_momentum=0.8,                  # Starting momentum for warmup
    warmup_bias_lr=0.1,                   # Learning rate for bias during warmup

    # Data augmentation — helps improve generalization
    hsv_h=0.015,                          # HSV hue augmentation
    hsv_s=0.4,                            # HSV saturation
    hsv_v=0.4,                            # HSV brightness
    degrees=10,                           # Random rotation
    translate=0.1,                        # Image translation
    scale=0.4,                            # Scaling (zoom in/out)
    shear=0.1,                            # Shearing
    perspective=0.0,                      # Perspective transformation
    flipud=0.1,                           # Vertical flip with 10% probability
    fliplr=0.5,                           # Horizontal flip with 50% probability
    mosaic=0.9,                           # Mosaic augmentation probability
    mixup=0.0,                            # Mixup disabled (not useful in this case)

    # Miscellaneous
    dropout=0.0,                          # No dropout
    cache=True,                           # Cache images in memory for faster training
    save=True,                            # Save checkpoints during training
    save_period=10,                       # Save model every 10 epochs
    val=True,                             # Run validation after each epoch
    device=0                              # Use GPU (0 = first CUDA device); use 'cpu' if no GPU
)

# Predict on a sample image
model.predict(
    source="/content/YOLOv8_Project/sample1.jpg",  # Replace with your own image path
    save=True,
    conf=0.5
)

#--------------------------------------------------------------------------------------------------
# Note : When the above code is runned once (the model is trained) and it can be commented to just see the output via best.pt file
# Use Ctrl + / to undo the coment above th dotted line (-----) to train the model

from ultralytics import YOLO
import cv2

# Load your trained model (change path to your best.pt)
model = YOLO(r"C:\Users\Atharv\Desktop\trained_model\best.pt")
# Open webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

# Optional: set resolution
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 prediction
    results = model(frame, conf=0.5)

    # Annotate results directly on the frame
    annotated_frame = results[0].plot()  # Automatically draws boxes and labels

    # Show the frame
    cv2.imshow("YOLOv8 Phone Detection", annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
