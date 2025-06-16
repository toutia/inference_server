import numpy as np
import time
import tritonclient.grpc as grpcclient
from tritonclient.grpc import InferInput
from PIL import Image
import cv2

import torch

img_path = "/home/touti/dev/DeepLabV3Plus-Pytorch/test_low_res/20250528_205520.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Setup
model_name = "deeplabv3plus"
server_url = "localhost:8001"  # gRPC port
input_name = "input"
output_name = "output"

# transform = T.Compose(
#     [
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )

# Connect to Triton server
client = grpcclient.InferenceServerClient(url=server_url)

# Prepare input tensor

# Load or create dummy input: shape (1, 3, 512, 512)
# Replace with your actual image preprocessing

# === Load and preprocess image ===
img = cv2.imread(img_path)  # BGR, HWC
img = cv2.resize(img, (512, 512))
img = img.astype(np.float32) / 255.0
img = img.transpose(2, 0, 1)  # CHW
img = np.expand_dims(img, axis=0)  # NCHW -> (1, 3, 512, 512)


# input_data = Image.open(img_path).convert("RGB")
# input_data = transform(input_data).unsqueeze(0)  # To tensor of NCHW
input_data = img


input_tensor = InferInput(input_name, input_data.shape, "FP32")
input_tensor.set_data_from_numpy(input_data)

# Warmup (optional)
client.infer(model_name, inputs=[input_tensor])

# Measure inference time
start_time = time.time()
response = client.infer(model_name, inputs=[input_tensor])
end_time = time.time()

# Process output
output = response.as_numpy(output_name)

mask = np.argmax(output[0], axis=0).astype(np.uint8)  # shape: (512, 512)


# === Save result ===
cv2.imwrite("segmentation_mask.png", mask * (255 // output.shape[1]))


# Report
print(f"Inference time: {(end_time - start_time) * 1000:.2f} ms")
print(f"Output shape: {output.shape}")
