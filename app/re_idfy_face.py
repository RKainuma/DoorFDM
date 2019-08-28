'''Get Face feature vecrots'''
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import platform

from openvino.inference_engine import IENetwork, IEPlugin
from face_utils import align_face


class FaceReIdentification():
    def __init__(self):
        if platform.system() == 'Windows':
            self.plugin = IEPlugin(device='CPU', plugin_dirs=None)
            model_xml = "./extension/IR/FP32/face-reidentification-retail-0095.xml"
            model_bin = "./extension/IR/FP32/face-reidentification-retail-0095.bin"
            self.cpu_extension = 'extension/cpu_extension.dll'
        elif platform.system() == 'Darwin':
            self.plugin = IEPlugin(device='CPU', plugin_dirs=None)
            model_xml = "./extension/IR/FP32/face-reidentification-retail-0095.xml"
            model_bin = "./extension/IR/FP32/face-reidentification-retail-0095.bin"
            self.cpu_extension = 'extension/libcpu_extension.dylib'
        else:
            self.plugin = IEPlugin(device='MYRIAD', plugin_dirs=None)
            model_xml = "./extension/IR/FP16/face-reidentification-retail-0095.xml"
            model_bin = "./extension/IR/FP16/face-reidentification-retail-0095.bin"
            self.cpu_extension = 'extension/libcpu_extension.dylib'

        net = IENetwork(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        self.plugin.add_cpu_extension(self.cpu_extension)
        self.exec_net = self.plugin.load(network=net, num_requests=2)
        self.facial_landmark = FacialLamdmark()

    def get_feature_vec(self, face_img):
        aligned_face = self.facial_landmark.turn_face(face_img)
        in_frame = cv2.resize(aligned_face, (self.w, self.h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
        self.exec_net.infer(inputs={self.input_blob: in_frame})
        res = self.exec_net.requests[0].outputs[self.out_blob]  # (1, 256, 1, 1)
        feature_vec = res.reshape(1, 256)

        return feature_vec[0]


class FacialLamdmark():
    def __init__(self):
        if platform.system() == 'Windows':
            self.plugin = IEPlugin(device='CPU', plugin_dirs=None)
            model_xml = "./extension/IR/FP32/landmarks-regression-retail-0009.xml"
            model_bin = "./extension/IR/FP32/landmarks-regression-retail-0009.bin"
            self.cpu_extension = 'extension/cpu_extension.dll'
        elif platform.system() == 'Darwin':
            self.plugin = IEPlugin(device='CPU', plugin_dirs=None)
            model_xml = "./extension/IR/FP32/landmarks-regression-retail-0009.xml"
            model_bin = "./extension/IR/FP32/landmarks-regression-retail-0009.bin"
            self.cpu_extension = './extension/libcpu_extension.dylib'
        else:
            self.plugin = IEPlugin(device='MYRIAD', plugin_dirs=None)
            model_xml = "./extension/IR/FP16/landmarks-regression-retail-0009.xml"
            model_bin = "./extension/IR/FP16/landmarks-regression-retail-0009.bin"
            self.cpu_extension = './extension/libcpu_extension.dylib'

        net = IENetwork(model=model_xml, weights=model_bin)
        self.input_blob = next(iter(net.inputs))
        self.out_blob = next(iter(net.outputs))
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        self.plugin.add_cpu_extension(self.cpu_extension)
        self.exec_net = self.plugin.load(network=net, num_requests=2)
        self.face_id = 0

    def turn_face(self, face_img):
        facial_landmarks = np.zeros((face_img.shape[2], 5, 2))
        in_frame = cv2.resize(face_img, (self.w, self.h))
        in_frame = in_frame.transpose((2, 0, 1))
        in_frame = in_frame.reshape((self.n, self.c, self.h, self.w))
        self.exec_net.start_async(request_id=0, inputs={self.input_blob: in_frame})

        if self.exec_net.requests[0].wait(-1) == 0:
            res = self.exec_net.requests[0].outputs[self.out_blob].reshape(1, 10)[0]

            lm_face = face_img.copy()
            for i in range(res.size // 2):
                normed_x = res[2 * i]
                normed_y = res[2 * i + 1]
                x_lm = lm_face.shape[1] * normed_x
                y_lm = lm_face.shape[0] * normed_y
                cv2.circle(lm_face, (int(x_lm), int(y_lm)), 1 + int(0.03 * lm_face.shape[1]), (255, 255, 0), -1)
                facial_landmarks[self.face_id][i] = (x_lm, y_lm)

            aligned_face = face_img.copy()
            aligned_face = align_face(aligned_face, facial_landmarks[self.face_id])

            return aligned_face
