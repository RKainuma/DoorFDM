#!/usr/bin/env python3
import cv2
import numpy as np
import math
import os

resize_prop = (720, 480)


class VideoCamera(object):
    def __init__(self, detections, no_v4l):

        self.detections = detections
        self.is_async_mode = False
        self.is_face_detection = True
        self.input_stream = 0
        # NOTE need to check os, Linux, Windows or Mac
        if no_v4l:
            self.cap = cv2.VideoCapture(self.input_stream)
        else:  # for Picamera, added VideoCaptureAPIs(cv2.CAP_V4L)
            try:
                self.cap = cv2.VideoCapture(self.input_stream, cv2.CAP_V4L)
            except:
                import traceback
                traceback.print_exc()
                print("\nPlease try to start with command line parameters using --no_v4l\n")
                os._exit(0)

        ret, self.frame = self.cap.read()
        cap_prop = self._get_cap_prop()

    def __del__(self):
        self.cap.release()

    def _get_cap_prop(self):

        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self):

        if self.is_async_mode:
            ret, next_frame = self.cap.read()
            if not ret:
                return None
            next_frame = cv2.resize(next_frame, resize_prop)
        else:
            ret, self.frame = self.cap.read()
            if not ret:
                return None
            self.frame = cv2.resize(self.frame, resize_prop)
            next_frame = None

        if self.is_face_detection:
            self.frame = self.detections.face_detection(self.frame, next_frame)

        if self.is_async_mode:
            self.frame = next_frame

        VideoCameraret, jpeg = cv2.imencode('1.jpg', self.frame)

        return jpeg.tostring()
