'''Process identify'''
# !/usr/bin/env python
# -*- coding: utf-8 -*-
from logging import getLogger, basicConfig, DEBUG, INFO, ERROR
from timeit import default_timer as timer
from queue import Queue
import numpy as np
import os
import sys
import cv2
import math
import platform
import glob
import time

from re_idfy_face import FaceReIdentification
from face_utils import cos_similarity
import detectors

logger = getLogger(__name__)
basicConfig(
    level=INFO,
    format="%(asctime)s %(levelname)s %(name)s %(funcName)s(): %(message)s")

FP32 = "extension/IR/FP32/"
FP16 = "extension/IR/FP16/"
model_fc_xml = "face-detection-retail-0004.xml"
model_hp_xml = "head-pose-estimation-adas-0001.xml"
face_reidfy = FaceReIdentification()


class Detectors(object):
    def __init__(self):
        if platform.system() == 'Darwin':
            self.devices = ['CPU', 'CPU']
        elif platform.system() == 'Windows':
            self.devices = ['CPU', 'CPU']
        else:
            self.devices = ['MYRIAD', 'MYRIAD']
        self.models = [None, None]
        self.plugin_dir = None
        if platform.system() == 'Darwin':
            self.cpu_extension = 'extension/libcpu_extension.dylib'
        else:
            self.cpu_extension = 'extension/cpu_extension.dll'
        self.prob_threshold = 0.3
        self.prob_threshold_face = 0.5
        self.is_async_mode = False
        self._load_detectors()

    def _load_detectors(self):
        device_fc, device_hp, = self.devices
        self.models = self._define_models()
        model_fc, model_hp = self.models
        cpu_extension = self.cpu_extension
        self.face_detectors = detectors.FaceDetection(
            device_fc, model_fc, cpu_extension, self.plugin_dir,
            self.prob_threshold_face, self.is_async_mode)

    def _define_models(self):
        device_fc, device_hp = self.devices
        model_fc, model_hp = self.models

        fp_path = FP32 if device_fc == "CPU" else FP16
        model_fc = fp_path + model_fc_xml if model_fc is None else model_fc
        fp_path = FP32 if device_hp == "CPU" else FP16
        model_hp = fp_path + model_hp_xml if model_hp is None else model_hp

        return [model_fc, model_hp]


class Detections(Detectors):
    def __init__(self):
        super().__init__()
        self.is_async_mode = False
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"
        self.prev_time = timer()
        self.time_stock = []
        self.interval = True

    def face_detection(self, frame, next_frame):
        start = time.time()
        color = (0, 255, 0)
        det_time = 0
        det_time_hp = 0
        det_time_txt = ""

        frame_h, frame_w = frame.shape[:2]
        is_face_analytics_enabled = True

        inf_start = timer()
        self.face_detectors.submit_req(frame, next_frame, self.is_async_mode)
        ret = self.face_detectors.wait()
        faces = self.face_detectors.get_results(self.is_async_mode)
        inf_end = timer()
        det_time = inf_end - inf_start

        face_count = faces.shape[2]
        det_time_txt = "face_cnt:{} face:{:.3f} ms ".format(face_count, det_time * 1000)

        if face_count > 1:
            is_face_async_mode = False
        else:
            is_face_async_mode = False

        face_id = 0
        face_w, face_h = 0, 0
        face_frame = None
        next_face_frame = None
        prev_box = None

        face_q = Queue()
        for face in faces[0][0]:
            face_q.put(face)

        if is_face_async_mode:
            face_count = face_count + 1

        for face_id in range(face_count):
            face_id = 0
            face_analytics = ""
            head_pose = ""

            if not face_q.empty():
                face = face_q.get()

            box = face[3:7] * np.array([frame_w, frame_h, frame_w, frame_h])
            xmin, ymin, xmax, ymax = box.astype("int")
            class_id = int(face[1])
            result = str(face_id) + " " + str(round(face[2] * 100, 1)) + '% '

            if xmin < 0 or ymin < 0:
                return frame

            if is_face_async_mode:
                next_face_frame = frame[ymin:ymax, xmin:xmax]
                if next_face_frame is None:
                    return frame
                if prev_box is not None:
                    xmin, ymin, xmax, ymax = prev_box.astype("int")
            else:
                # Face Identification
                face_frame = frame[ymin:ymax, xmin:xmax]
                ret_reid = face_reidfy.get_feature_vec(face_frame)
                face_pts_list = glob.glob('./face_pts/*')
                min_diff = 0
                threshold = 0.7
                for face_pts_file in face_pts_list:
                    faces_diff, nickname = cos_similarity(ret_reid, face_pts_file)
                    if (faces_diff > threshold) and (faces_diff > min_diff):
                        min_diff = faces_diff
                        identified_person = nickname
                    else:
                        pass
                if (min_diff > threshold) and (self.interval is True):
                    print('Found \033[92m{}\033[0m and Condfidence is \033[92m{}\033[0m'.format(identified_person, min_diff))
                    self.interval = False
                else:
                    # print('Face Not Identified')
                    pass
            if face_frame is not None:
                face_w, face_h = face_frame.shape[:2]
                if face_w == 0 or face_h == 0:
                    logger.error(
                        "Unexpected shape of face frame. face_frame.shape:{} {}".
                        format(face_h, face_w))
                    return frame
        end = time.time()
        elapsed = end - start
        self.time_stock.append(elapsed)
        if sum(self.time_stock) > 4:
            self.interval = True
            self.time_stock.clear()
            print('Interval END')
        else:
            pass

        return frame
