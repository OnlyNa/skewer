from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO("yolov8n.yaml")

# Load a pretrained YOLO model (recommended for training)
model.load('yolov8n.pt')

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data="./datasets/skewer.yaml", epochs=30)


# Evaluate the model's performance on the validation set
# results = model.val()
# print('val results:', results)
# Perform object detection on an image using the model
# results = model("https://ultralytics.com/images/bus.jpg")
# print(f'results 1:{results}')
# results = model.predict("test.jpg",save=True)
# print(f'results 2:{results}')

# Export the model to ONNX format
# success = model.export(format="onnx")
