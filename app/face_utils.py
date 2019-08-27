#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2

def align_face(face_frame, landmarks):
    left_eye, right_eye, tip_of_nose, left_lip, right_lip = landmarks

    # compute the angle between the eye centroids
    dy = right_eye[1] - left_eye[1]  # right eye, left eye Y
    dx = right_eye[0] - left_eye[0]  # right eye, left eye X
    angle = np.arctan2(dy, dx) * 180 / np.pi

    ## center of face_frame
    center = (face_frame.shape[0] // 2, face_frame.shape[1] // 2)
    h, w, c = face_frame.shape

    # grab the rotation matrix for rotating and scaling the face
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    aligned_face = cv2.warpAffine(face_frame, M, (w, h))

    return aligned_face

def cos_similarity(aligned_face):
    with open('face_pts/kai.reid', 'rb') as f:
        X = np.load(f)
    Y = aligned_face

    return np.dot(X, Y)/(np.linalg.norm(X) * np.linalg.norm(Y, axis=0))

