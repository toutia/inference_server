from ultralytics import YOLO

# Load the Triton Server model
model = YOLO("http://localhost:8000/yolo11m", task='detect')

# Run inference on the server
results = model("/home/touti/dev/triton_manager/triton_examples/bus.jpg")

for result in results:
    print(result.boxes)