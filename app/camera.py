'''Camera setting and get frames with OpenCV, VideoCapture'''
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
import os


class VideoCamera(object):
    def __init__(self, detections, no_v4l):
        self.detections = detections
        self.is_async_mode = False
        self.is_face_detection = True
        self.input_stream = 0
        self.resize_prop = (720, 480)
        if no_v4l:
            self.cap = cv2.VideoCapture(self.input_stream)
        else:
            try:
                self.cap = cv2.VideoCapture(self.input_stream, cv2.CAP_V4L)
            except:
                import traceback
                traceback.print_exc()
                print("\nPlease try to start with command line parameters using --no_v4l\n")
                os._exit(0)

        ret, self.frame = self.cap.read()

    def __del__(self):
        self.cap.release()

    def _get_cap_prop(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        if self.is_async_mode:
            ret, next_frame = self.cap.read()
            if not ret:
                return None
            next_frame = cv2.resize(next_frame, self.resize_prop)
        else:
            ret, self.frame = self.cap.read()
            if not ret:
                return None
            self.frame = cv2.resize(self.frame, self.resize_prop)
            next_frame = None

        if self.is_face_detection:
            self.frame = self.detections.face_detection(self.frame, next_frame)

        if self.is_async_mode:
            self.frame = next_frame

        VideoCameraret, jpeg = cv2.imencode('1.jpg', self.frame)

        return jpeg.tostring()
