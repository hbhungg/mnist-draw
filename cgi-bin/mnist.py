#!/usr/bin/env python3
"""
CGI script that accepts image urls and feeds them into a ML classifier. Results
are returned in JSON format. 
"""

import io
import json
import sys
import os
import re
import base64
import numpy as np
from PIL import Image
import torch
import torchvision

from model import Model

# Default output
res = {"result": 0,
       "data": [], 
       "error": '',
       "debug": []}

device = torch.device("cpu")
model = Model()
model.load_state_dict(torch.load("./cgi-bin/results/model.pth", map_location=device))
model.eval()

transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
  (0.1307,), (0.3081,))
])

try:
  # Get post data
  if os.environ["REQUEST_METHOD"] == "POST":
    data = sys.stdin.read(int(os.environ["CONTENT_LENGTH"]))

    # Convert data url to numpy array
    img_str = re.search(r'base64,(.*)', data).group(1)
    image_bytes = io.BytesIO(base64.b64decode(img_str))
    im = Image.open(image_bytes).convert("L")

    arr = transform(im)

    # Predict class
    predictions = model(arr)
    predictions = torch.nn.functional.softmax(predictions, dim=1).tolist()[0]
    #prediction = torch.nn.functional.softmax(prediction).data.max(1, keepdim=True)[1]
    #predictions = [1 if i == prediction else 0 for i in range(0, 10)]

    # Return label data
    res['result'] = 1
    res['data'] = predictions

except Exception as e:
  # Return error data
  res['error'] = str(e)

# Print JSON response
print("Content-type: application/json")
print("") 
print(json.dumps(res))


