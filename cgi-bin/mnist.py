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
from PIL import Image, ImageOps
import torch
import torchvision
import math

from model import Model

# Default output
res = {"result": 0,
       "data": [], 
       "error": '',
       "debug": []}

model = Model()
model.load_state_dict(torch.load("./cgi-bin/results/model.pth"))
model = model.eval()

transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
  (0.1307,), (0.3081,))
])

#try:
  # Get post data
if os.environ["REQUEST_METHOD"] == "POST":
  data = sys.stdin.read(int(os.environ["CONTENT_LENGTH"]))

  # Convert data url to numpy array
  img_str = re.search(r'base64,(.*)', data).group(1)
  image_bytes = io.BytesIO(base64.b64decode(img_str))
  im = Image.open(image_bytes).convert("L")

  # Resize and invert
  im = im.resize((28, 28))
  im = ImageOps.invert(im)

  im.save("./test.jpg")

  arr = transform(im)
  arr = torch.unsqueeze(arr, 0) 

  torch.save(arr, 'tensor.pt')

  # Predict class
  predictions = model(arr).tolist()[0]
  predictions = [math.exp(i) for i in predictions] 

  # Return label data
  res['result'] = 1
  res['data'] = predictions

#except Exception as e:
#  # Return error data
#  res['error'] = str(e)
#  print(e)

# Print JSON response
print("Content-type: application/json")
print("") 
print(json.dumps(res))


