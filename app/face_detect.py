import requests
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import os
import sys
import math
from openvino.inference_engine import IENetwork, IEPlugin

def face_detction(image_path):
  # Read Image
  frame = cv2.imread(image_path)
  # resize image with keeping frame width
  scale = 640 / frame.shape[1]
  frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
  frame_h, frame_w = frame.shape[:2]
  init_frame = frame.copy()
  # Face Detection
  # 1. Plugin initialization for specified device and load extensions library if specified
  device = "CPU"
  fp_path = "./extension/IR/FP32/" if device == "CPU" else "./extension/IR/FP16/"
  plugin = IEPlugin(device=device, plugin_dirs=None)
  if device == "CPU":
      plugin.add_cpu_extension("./extension/libcpu_extension.dylib")

  # 2.Read IR
  model_xml = fp_path + "face-detection-adas-0001.xml"
  model_bin = os.path.splitext(model_xml)[0] + ".bin"
  net = IENetwork(model=model_xml, weights=model_bin)

  # 3. Configure input & output
  input_blob = next(iter(net.inputs))
  out_blob = next(iter(net.outputs))
  n, c, h, w = net.inputs[input_blob].shape
  # 4. Load Model
  exec_net = plugin.load(network=net, num_requests=2)

  # 5. Create Async Request
  in_frame = cv2.resize(frame, (w, h))
  in_frame = in_frame.transpose((2, 0, 1))
  in_frame = in_frame.reshape((n, c, h, w))
  exec_net.start_async(request_id=0, inputs={input_blob: in_frame}) # res's shape: [1, 1, 200, 7]

  # 6. Receive Async Request
  if exec_net.requests[0].wait(-1) == 0:
      res = exec_net.requests[0].outputs[out_blob]
      faces = res[0][:, np.where(res[0][0][:, 2] > 0.5)] # prob threshold : 0.5


  # 7. draw faces
  croped_faces = []
  frame = init_frame.copy()
  for face in faces[0][0]:
      box = face[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
      (xmin, ymin, xmax, ymax) = box.astype("int")
      croped_face = frame[ymin:ymax, xmin:xmax]
      croped_faces.append(croped_face)

  return croped_faces
