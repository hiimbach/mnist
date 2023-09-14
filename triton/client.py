import os 
import sys
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.append(ROOT)

import tritonclient.http as httpclient
from utils.cv2_to_np import cv2_to_array
import numpy as np
import cv2

# Set up connection with Triton Inference Server
client = httpclient.InferenceServerClient(url="192.168.65.4:8000/")

# Preprocess Images
frame = cv2.imread("data/5.1.png", 0)
arr = cv2_to_array(frame)

# Normalize and convert from FP64 -> FP32
# norm = (arr-mean)/std
arr = (arr-0.1307)/0.3081

# Because triton requires shape [B, C, H, W]
arr = np.expand_dims(arr.astype(np.float32), axis=0) 
arr = np.expand_dims(arr, axis=0) 

# Names the input and output layers of model
inputs = httpclient.InferInput("input", arr.shape, datatype="FP32")
inputs.set_data_from_numpy(arr)
outputs = httpclient.InferRequestedOutput("output", binary_data=True, class_count=10)

# Querying the server
results = client.infer(model_name="mnist", inputs=[inputs], outputs=[outputs])
inference_output = results.as_numpy('output').astype(str)

print(inference_output[0][0][-1])
# import ipdb; ipdb.set_trace()

