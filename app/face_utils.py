'''Utils methods when used by main app'''
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from functools import partial
import cv2
import platform
import os
import pathlib

from openvino.inference_engine import IENetwork, IEPlugin

np.load = partial(np.load, allow_pickle=True) 


def align_face(face_frame, landmarks):
    left_eye, right_eye, tip_of_nose, left_lip, right_lip = landmarks

    # compute the angle between the eye centroids
    dy = right_eye[1] - left_eye[1]  # right eye, left eye Y
    dx = right_eye[0] - left_eye[0]  # right eye, left eye X
    angle = np.arctan2(dy, dx) * 180 / np.pi

    # center of face_frame
    center = (face_frame.shape[0] // 2, face_frame.shape[1] // 2)
    h, w, c = face_frame.shape

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_face = cv2.warpAffine(face_frame, M, (w, h))

    return aligned_face


def cos_similarity(aligned_face, face_pts_file):
    with open(face_pts_file, 'rb') as f:
        X = np.load(f)
    Y = aligned_face
    nickname = face_pts_file.lstrip('./face_pts/').replace(".reid", '')

    return np.dot(X, Y)/(np.linalg.norm(X) * np.linalg.norm(Y, axis=0)), nickname


def face_detction(image_path):
    frame = cv2.imread(image_path)
    scale = 640 / frame.shape[1]
    frame = cv2.resize(frame, dsize=None, fx=scale, fy=scale)
    frame_h, frame_w = frame.shape[:2]
    init_frame = frame.copy()

    # Face Detection
    # 1. Plugin initialization for specified device and load extensions library if specified
    if platform.system() == 'Windows':
        device = "CPU"
        extension_path = 'extension/cpu_extension.dll'
    elif platform.system() == 'Darwin':
        device = "CPU"
        extension_path = 'extension/libcpu_extension.dylib'
    else:
        device = 'MYRIAD'
        extension_path = 'extension/libcpu_extension.dylib'

    extension_path = pathlib.Path(extension_path)
    ab_extension_path = str(extension_path.resolve())
    fp_path = "./extension/IR/FP32/" if device == "CPU" else "./extension/IR/FP16/"
    plugin = IEPlugin(device=device, plugin_dirs=None)

    if platform.system() == 'Windows':
        plugin.add_cpu_extension(ab_extension_path)
    else:
        plugin.add_cpu_extension(ab_extension_path)

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
    exec_net.start_async(request_id=0, inputs={input_blob: in_frame})  # res's shape: [1, 1, 200, 7]

    # 6. Receive Async Request
    if exec_net.requests[0].wait(-1) == 0:
        res = exec_net.requests[0].outputs[out_blob]
        faces = res[0][:, np.where(res[0][0][:, 2] > 0.5)]  # prob threshold : 0.5

    # 7. draw faces
    croped_faces = []
    frame = init_frame.copy()
    for face in faces[0][0]:
        box = face[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
        (xmin, ymin, xmax, ymax) = box.astype("int")
        croped_face = frame[ymin:ymax, xmin:xmax]
        croped_faces.append(croped_face)

    return croped_faces
